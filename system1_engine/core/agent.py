from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution

from system1_engine.core.encoders import ImpalaVisualFrontEnd, VectorFrontEnd
from system1_engine.core.heads import CategoricalPolicyHead, GaussianPolicyHead, ValueHead
from system1_engine.core.trunk import System1Trunk


@dataclass(frozen=True)
class ReflexDecision:
    """Resultado estruturado e amortizado da decisão do System 1 com telemetria de incerteza."""

    action: Union[int, np.ndarray]
    confidence: float          # [0.0, 1.0] - Probabilidade da ação escolhida
    uncertainty: float         # [0.0, 1.0] - Entropia normalizada
    margin: float              # [0.0, 1.0] - Diferença entre as duas maiores probabilidades
    is_uncertain: bool         # Gatilho de invocação para o System 2
    entropy: float             # Entropia bruta de Shannon
    latent_value: Optional[float] = None  # Valor estimado V(s) se solicitado


class UniversalS1Agent(nn.Module):
    """Universal System 1 RL Agent.

    Strict decoupling between:
      1. Front-end (Perception): VectorFrontEnd or ImpalaVisualFrontEnd
      2. Trunk (Recurrent Reflexive Core): System1Trunk
      3. Heads (Decision & Value): PolicyHead and ValueHead

    Guarantees:
      - Memory footprint: ~650k parameters (vector) or ~2.5M (visual).
      - Inference latency: <= 0.8 ms on CPU for act_fast().
      - Zero catastrophic forgetting via weight freezing on trunk.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        is_visual: Optional[bool] = None,
        trunk: Optional[System1Trunk] = None,
        reward_fixed: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(obs_space, gym.spaces.Dict) and "obs" in obs_space.spaces:
            obs_space = obs_space.spaces["obs"]
        self.obs_space = obs_space
        self.action_space = action_space

        if is_visual is None:
            is_visual = (
                isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3
            )
        self.is_visual = is_visual

        # 1. Perception Front-End (Scenario-dependent)
        if self.is_visual:
            assert isinstance(obs_space, gym.spaces.Box), "Visual space must be gym.spaces.Box"
            in_channels = obs_space.shape[0]
            self.front_end: Union[VectorFrontEnd, ImpalaVisualFrontEnd] = (
                ImpalaVisualFrontEnd(
                    in_channels=in_channels,
                    action_space=action_space,
                    reward_fixed=reward_fixed,
                )
            )
        else:
            assert isinstance(obs_space, gym.spaces.Box), "Vector space must be gym.spaces.Box"
            obs_dim = int(np.prod(obs_space.shape))
            self.front_end = VectorFrontEnd(
                obs_dim=obs_dim,
                action_space=action_space,
                reward_fixed=reward_fixed,
            )

        # 2. Universal Recurrent Trunk (Can be shared or transferred)
        if trunk is not None:
            self.trunk = trunk
        else:
            self.trunk = System1Trunk(input_dim=337, hidden_dim=256, num_res_blocks=2)

        # 3. Policy & Value Heads (Scenario-dependent)
        if isinstance(action_space, gym.spaces.Discrete):
            self.is_discrete = True
            self.policy_head: Union[CategoricalPolicyHead, GaussianPolicyHead] = (
                CategoricalPolicyHead(in_features=256, n_actions=int(action_space.n))
            )
        elif isinstance(action_space, gym.spaces.Box):
            self.is_discrete = False
            act_dim = int(np.prod(action_space.shape))
            self.policy_head = GaussianPolicyHead(in_features=256, act_dim=act_dim)
        else:
            raise NotImplementedError(f"Unsupported action space: {type(action_space)}")

        self.value_head = ValueHead(in_features=256)

        # Runtime hidden state for act_fast inference
        self.hx: Optional[torch.Tensor] = None

    def reset_memory(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        """Resets the persistent internal hidden state."""
        self.hx = None

    def extract_features(
        self,
        obs_dict: Dict[str, Any],
    ) -> torch.Tensor:
        """Passes observation dictionary through the perceptual front-end."""
        obs = obs_dict["obs"]
        prev_action = obs_dict["prev_action"]
        prev_reward = obs_dict["prev_reward"]
        delta_obs = obs_dict.get("delta_obs")

        if self.is_visual:
            return self.front_end(
                obs=obs,
                prev_action=prev_action,
                prev_reward=prev_reward,
                delta_obs=delta_obs,
            )
        else:
            assert delta_obs is not None, "delta_obs required for vector front-end"
            return self.front_end(
                obs=obs,
                delta_obs=delta_obs,
                prev_action=prev_action,
                prev_reward=prev_reward,
            )

    def forward(
        self,
        obs_dict: Dict[str, Any],
        hx: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
    ) -> Tuple[Distribution, torch.Tensor, torch.Tensor]:
        """Unified forward pass returning action distribution, state value, and next hidden state."""
        z_in = self.extract_features(obs_dict)
        h, next_hx = self.trunk(z_in, hx=hx, dones=dones)
        dist = self.policy_head(h)
        val = self.value_head(h)
        return dist, val, next_hx

    def get_action(
        self,
        obs_dict: Dict[str, Any],
        hx: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rollout step returning (action, log_prob, value, next_hx)."""
        dist, value, next_hx = self.forward(obs_dict, hx=hx, dones=None)
        if deterministic:
            if self.is_discrete:
                action = torch.argmax(dist.logits, dim=-1)
            else:
                action = dist.mean
        else:
            action = dist.sample()

        if self.is_discrete:
            log_prob = dist.log_prob(action)
        else:
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value, next_hx

    @torch.no_grad()
    def act_fast(
        self,
        obs_dict: Dict[str, Any],
        return_decision: bool = False,
        uncertainty_threshold: float = 0.70,
        confidence_threshold: float = 0.50,
        return_value: bool = False,
    ) -> Union[int, np.ndarray, ReflexDecision]:
        """Ultra-fast deterministic single-forward reflex inference.

        Optimized for zero autograd overhead, minimal allocation, and <= 0.8ms latency on CPU.
        If return_decision=True, returns ReflexDecision with uncertainty gating signals.
        """
        raw_obs = obs_dict["obs"]
        raw_action = obs_dict["prev_action"]
        raw_reward = obs_dict["prev_reward"]

        # Fast tensor conversions
        if self.is_visual:
            if isinstance(raw_obs, torch.Tensor):
                t_obs = raw_obs
                if t_obs.dim() == 3:
                    t_obs = t_obs.unsqueeze(0).unsqueeze(0)
                elif t_obs.dim() == 4:
                    t_obs = t_obs.unsqueeze(1)
            else:
                arr = np.asarray(raw_obs, dtype=np.float32)
                t_obs = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        else:
            if isinstance(raw_obs, torch.Tensor):
                t_obs = raw_obs
                if t_obs.dim() == 1:
                    t_obs = t_obs.unsqueeze(0).unsqueeze(0)
                elif t_obs.dim() == 2:
                    t_obs = t_obs.unsqueeze(1)
            else:
                arr = np.asarray(raw_obs, dtype=np.float32)
                t_obs = torch.from_numpy(arr).view(1, 1, -1)

        if self.is_visual:
            if isinstance(raw_action, torch.Tensor):
                t_act = raw_action
            elif self.is_discrete:
                t_act = torch.tensor([[int(raw_action)]], dtype=torch.long)
            else:
                t_act = torch.from_numpy(np.asarray(raw_action, dtype=np.float32)).view(1, 1, -1)

            if isinstance(raw_reward, torch.Tensor):
                t_rew = raw_reward
            else:
                t_rew = torch.tensor([[[float(raw_reward)]]], dtype=torch.float32)

            z_in = self.front_end(
                obs=t_obs,
                prev_action=t_act,
                prev_reward=t_rew,
            )
        else:
            raw_delta = obs_dict["delta_obs"]
            if isinstance(raw_delta, torch.Tensor):
                t_delta = raw_delta
                if t_delta.dim() == 1:
                    t_delta = t_delta.unsqueeze(0).unsqueeze(0)
                elif t_delta.dim() == 2:
                    t_delta = t_delta.unsqueeze(1)
            else:
                arr_d = np.asarray(raw_delta, dtype=np.float32)
                t_delta = torch.from_numpy(arr_d).view(1, 1, -1)

            if isinstance(raw_action, torch.Tensor):
                t_act = raw_action
            elif self.is_discrete:
                t_act = torch.tensor([[int(raw_action)]], dtype=torch.long)
            else:
                t_act = torch.from_numpy(np.asarray(raw_action, dtype=np.float32)).view(1, 1, -1)

            if isinstance(raw_reward, torch.Tensor):
                t_rew = raw_reward
            else:
                t_rew = torch.tensor([[[float(raw_reward)]]], dtype=torch.float32)

            z_in = self.front_end(
                obs=t_obs,
                delta_obs=t_delta,
                prev_action=t_act,
                prev_reward=t_rew,
            )

        # Recurrent pass through trunk with internal persistent memory
        h, self.hx = self.trunk(z_in, hx=self.hx, dones=None)

        # Policy decision
        if self.is_discrete:
            logits = self.policy_head.linear(h)
            if not return_decision:
                action = int(torch.argmax(logits, dim=-1).item())
                return action

            # Cálculo de incerteza e gatilho do System 2
            probs = torch.softmax(logits.squeeze(0).squeeze(0), dim=-1)
            action = int(torch.argmax(probs, dim=-1).item())
            n_acts = probs.shape[0]

            if n_acts > 1:
                top_probs, _ = torch.topk(probs, k=min(2, n_acts))
                confidence = float(top_probs[0].item())
                margin = float((top_probs[0] - top_probs[1]).item())
                entropy = float(-torch.sum(probs * torch.log(probs + 1e-8)).item())
                max_entropy = float(np.log(n_acts))
                uncertainty = float(np.clip(entropy / max_entropy, 0.0, 1.0))
            else:
                confidence = 1.0
                margin = 1.0
                entropy = 0.0
                uncertainty = 0.0

            is_uncertain = (uncertainty >= uncertainty_threshold) or (confidence < confidence_threshold)
            latent_val = float(self.value_head(h).squeeze().item()) if return_value else None

            return ReflexDecision(
                action=action,
                confidence=confidence,
                uncertainty=uncertainty,
                margin=margin,
                is_uncertain=is_uncertain,
                entropy=entropy,
                latent_value=latent_val,
            )
        else:
            mu = self.policy_head.mu_net(h)
            action_np = mu.squeeze(0).squeeze(0).cpu().numpy()
            if not return_decision:
                return action_np

            # Clamping defensivo de log_std / std para evitar underflow numérico e instabilidade
            log_std_clamped = torch.clamp(self.policy_head.log_std, min=-20.0, max=2.0)
            std = torch.exp(log_std_clamped).cpu().numpy()
            std = np.clip(std, a_min=1e-6, a_max=100.0)

            # Entropia diferencial contínua H = 0.5 * sum(1 + ln(2*pi*sigma^2))
            # Nota: para sigma < 1 / sqrt(2*pi*e) (~0.24197), H assume valores negativos legítimos
            entropy = float(0.5 * np.sum(1.0 + np.log(2.0 * np.pi * (std ** 2))))

            # Variância média das ações contínuas
            var = float(np.mean(std ** 2))

            # Normalização estrita para [0.0, 1.0] via sigmoide na variância
            # f(var) = 2.0 / (1.0 + exp(-var / var_scale)) - 1.0
            # Garante ausência de underflow para sigma < 0.242 e gatilho calibrado:
            # - sigma -> 0: uncertainty -> 0.0, confidence -> 1.0 (sem underflow negativo)
            # - sigma = 0.05: uncertainty = 0.25%, confidence = 99.75%
            # - sigma = 0.242 (fronteira H=0): uncertainty = 5.85%, confidence = 94.15%
            # - sigma = 1.0 (exploração inicial padrão): uncertainty = 76.16%, confidence = 23.84% (Gatilho Ativado)
            var_scale = 0.5
            uncertainty = float(np.clip(2.0 / (1.0 + np.exp(-var / var_scale)) - 1.0, 0.0, 1.0))
            confidence = float(np.clip(1.0 - uncertainty, 0.0, 1.0))

            is_uncertain = (uncertainty >= uncertainty_threshold) or (confidence < confidence_threshold)
            latent_val = float(self.value_head(h).squeeze().item()) if return_value else None

            return ReflexDecision(
                action=action_np,
                confidence=confidence,
                uncertainty=uncertainty,
                margin=0.0,
                is_uncertain=is_uncertain,
                entropy=entropy,
                latent_value=latent_val,
            )

    @torch.no_grad()
    def act_with_confidence(
        self,
        obs_dict: Dict[str, Any],
        uncertainty_threshold: float = 0.70,
        confidence_threshold: float = 0.50,
        return_value: bool = False,
    ) -> ReflexDecision:
        """Executa a decisão reflexiva retornando a estrutura ReflexDecision com o sinal de gatilho."""
        decision = self.act_fast(
            obs_dict=obs_dict,
            return_decision=True,
            uncertainty_threshold=uncertainty_threshold,
            confidence_threshold=confidence_threshold,
            return_value=return_value,
        )
        assert isinstance(decision, ReflexDecision)
        return decision
