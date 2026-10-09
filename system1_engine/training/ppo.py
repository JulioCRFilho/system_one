from collections import deque
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.telemetry.tracker import LiveStatsTracker
from system1_engine.training.buffer import RecurrentRolloutBuffer


class RecurrentPPOTrainer:
    """Pure PyTorch Recurrent PPO Trainer with GRU hidden state preservation and BPTT."""

    def __init__(
        self,
        agent: UniversalS1Agent,
        env: UniversalS1Wrapper,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.05,
        entropy_coef: float = 0.001,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        rollout_steps: int = 1024,
        chunk_length: int = 16,
        chunk_batch_size: int = 16,
        device: Union[str, torch.device] = "cpu",
        tracker: Optional[LiveStatsTracker] = None,
        exploration_scale: Union[float, str] = 1.0,
    ) -> None:
        self.agent = agent.to(device)
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.rollout_steps = rollout_steps
        self.chunk_length = chunk_length
        self.chunk_batch_size = chunk_batch_size
        self.device = torch.device(device)
        self.tracker = tracker

        # Suporte a exploração automática / homeostática no treino PPO
        if isinstance(exploration_scale, str) and exploration_scale.lower() in ("auto", "adaptive", "homeostatic"):
            self.is_auto_exploration = True
            self.exploration_scale = 0.80
            self._adaptive_scale = 0.80
            self._reward_ema = 0.0
            self._stagnation_count = 0
            self._recent_actions: deque[int] = deque(maxlen=6)
        else:
            self.is_auto_exploration = False
            self.exploration_scale = float(np.clip(float(exploration_scale), 0.05, 1.0))
            self._adaptive_scale = self.exploration_scale
            self._reward_ema = 0.0
            self._stagnation_count = 0
            self._recent_actions = deque(maxlen=6)

        # Optimize only parameters that require grad (respects frozen trunk)
        trainable_params = [p for p in self.agent.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "No trainable parameters found!"
        self.optimizer = optim.Adam(trainable_params, lr=learning_rate, eps=1e-5)
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None

        # Rollout buffer
        is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        obs_shape = env.observation_space["obs"].shape
        act_shape = () if is_discrete else env.action_space.shape

        self.buffer = RecurrentRolloutBuffer(
            buffer_size=rollout_steps,
            chunk_length=chunk_length,
            is_visual=env.is_visual,
            is_discrete=is_discrete,
            obs_shape=obs_shape,
            action_shape=act_shape,
            device=self.device,
        )

        # Buffers de tensores pré-alocados no dispositivo para eliminar overhead de heap e alocações repetidas
        self._buf_obs = torch.zeros((1, 1, *obs_shape), dtype=torch.float32, device=self.device)
        if is_discrete:
            self._buf_act = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        else:
            self._buf_act = torch.zeros((1, 1, *act_shape), dtype=torch.float32, device=self.device)
        self._buf_rew = torch.zeros((1, 1, 1), dtype=torch.float32, device=self.device)
        if not env.is_visual:
            self._buf_delta: Optional[torch.Tensor] = torch.zeros((1, 1, *obs_shape), dtype=torch.float32, device=self.device)
        else:
            self._buf_delta = None

        self._rollout_input_dict: Dict[str, Any] = {
            "obs": self._buf_obs,
            "prev_action": self._buf_act,
            "prev_reward": self._buf_rew,
        }
        if self._buf_delta is not None:
            self._rollout_input_dict["delta_obs"] = self._buf_delta

        # Persistent episode statistics across rollout batches
        self.episode_returns: deque[float] = deque(maxlen=20)
        self.curr_ep_return: float = 0.0
        self.total_steps: int = 0

    def _prepare_rollout_input(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Copia os dados do dicionário de observação diretamente para os tensores pré-alocados."""
        self._buf_obs[0, 0].copy_(torch.as_tensor(obs_dict["obs"], dtype=torch.float32))
        prev_act_raw = obs_dict["prev_action"]
        if self.agent.is_discrete:
            self._buf_act[0, 0] = int(prev_act_raw)
        else:
            self._buf_act[0, 0].copy_(torch.as_tensor(prev_act_raw, dtype=torch.float32))
        self._buf_rew[0, 0, 0] = float(obs_dict["prev_reward"])
        if self._buf_delta is not None and "delta_obs" in obs_dict and obs_dict["delta_obs"] is not None:
            self._buf_delta[0, 0].copy_(torch.as_tensor(obs_dict["delta_obs"], dtype=torch.float32))
        return self._rollout_input_dict

    def collect_rollouts(
        self,
        current_obs_dict: Dict[str, Any],
        current_hx: Optional[torch.Tensor],
        episode_start: bool,
        step_callback: Optional[Callable[[], None]] = None,
    ) -> Tuple[Dict[str, Any], Optional[torch.Tensor], bool, float]:
        """Collects rollout_steps interactions in the environment."""
        self.buffer.reset()
        self.agent.eval()

        with torch.no_grad():
            for _ in range(self.rollout_steps):
                input_dict = self._prepare_rollout_input(current_obs_dict)

                # Step agent with latency tracking
                t0 = time.perf_counter_ns()
                dist, value, next_hx = self.agent.forward(input_dict, hx=current_hx, dones=None)

                # Escala de exploração / homeostase dinâmica
                if self.is_auto_exploration:
                    curr_scale = float(self._adaptive_scale)
                else:
                    curr_scale = float(self.exploration_scale)

                if self.agent.is_discrete:
                    logits = dist.logits
                    # Quebra de ciclos (Cycle Breaking) sob estagnação no modo automático
                    if self.is_auto_exploration and self._stagnation_count >= 3 and len(self._recent_actions) >= 3:
                        if self._recent_actions[-1] == self._recent_actions[-3]:
                            avoid_act = self._recent_actions[-2]
                            penalized_logits = logits.clone()
                            penalized_logits[..., avoid_act] -= 3.0
                            logits = penalized_logits

                    if curr_scale != 1.0:
                        scaled_logits = logits / curr_scale
                        dist_sample = torch.distributions.Categorical(logits=scaled_logits)
                    elif self.is_auto_exploration and self._stagnation_count >= 3:
                        dist_sample = torch.distributions.Categorical(logits=logits)
                    else:
                        dist_sample = dist
                    action = dist_sample.sample()
                    log_prob = dist_sample.log_prob(action)
                    env_action = int(action.item())
                    if self.is_auto_exploration:
                        self._recent_actions.append(env_action)
                else:
                    if curr_scale != 1.0:
                        scaled_std = dist.scale * curr_scale
                        dist_sample = torch.distributions.Normal(loc=dist.mean, scale=scaled_std)
                    else:
                        dist_sample = dist
                    action = dist_sample.sample()
                    log_prob = dist_sample.log_prob(action).sum(dim=-1)
                    env_action = action.squeeze(0).squeeze(0).cpu().numpy()
                    if isinstance(self.env.action_space, gym.spaces.Box):
                        env_action = np.clip(env_action, self.env.action_space.low, self.env.action_space.high)
                lat_us = (time.perf_counter_ns() - t0) / 1000.0

                if self.tracker is not None:
                    if self.agent.is_discrete:
                        probs = dist.probs.squeeze(0).squeeze(0)
                        n_acts = probs.shape[-1]
                        top_probs, _ = torch.topk(probs, k=min(2, n_acts))
                        conf = float(top_probs[0].item())
                        ent = float(-torch.sum(probs * torch.log(probs + 1e-8)).item())
                        max_ent = float(np.log(n_acts)) if n_acts > 1 else 1.0
                        unc = float(np.clip(ent / max_ent, 0.0, 1.0))
                    else:
                        std = dist.scale.squeeze(0).squeeze(0).cpu().numpy()
                        std = np.clip(std, a_min=1e-6, a_max=100.0)
                        ent = float(0.5 * np.sum(1.0 + np.log(2.0 * np.pi * (std ** 2))))
                        var = float(np.mean(std ** 2))
                        unc = float(np.clip(2.0 / (1.0 + np.exp(-var / 0.5)) - 1.0, 0.0, 1.0))
                        conf = float(np.clip(1.0 - unc, 0.0, 1.0))

                    self.tracker.record_inference(
                        latency_us=lat_us,
                        uncertainty=unc,
                        confidence=conf,
                        entropy=ent,
                        calibration=curr_scale,
                    )

                # Environment step
                next_obs_dict, reward, terminated, truncated, env_info = self.env.step(env_action)
                done = terminated or truncated

                # Auto-correção homeostática por momentum de recompensa
                if self.is_auto_exploration:
                    rew_f = float(reward)
                    delta_r = rew_f - self._reward_ema
                    self._reward_ema = 0.9 * self._reward_ema + 0.1 * rew_f

                    if rew_f > 0.01 or delta_r > 0.01:
                        # Progresso / recompensa: resfria para buscar o equilíbrio e estabilidade das decisões
                        self._stagnation_count = 0
                        self._adaptive_scale = max(0.15, self._adaptive_scale - 0.02)
                    else:
                        # Estagnação: aquece gradualmente para auto-correção e escape de armadilhas
                        self._stagnation_count += 1
                        if self._stagnation_count >= 3:
                            self._adaptive_scale = min(1.0, self._adaptive_scale + 0.01)

                if self.tracker is not None:
                    self.tracker.record_env_step(reward=reward, done=done, info=env_info)

                self.curr_ep_return += reward
                self.total_steps += 1

                if step_callback is not None:
                    step_callback()

                # Save transition into buffer: 'terminated' flags true terminal failure for GAE
                self.buffer.add(
                    obs_dict=current_obs_dict,
                    action=action.squeeze(0).squeeze(0),
                    reward=reward,
                    done=terminated,
                    episode_start=episode_start,
                    value=value.squeeze(0).squeeze(0),
                    log_prob=log_prob.squeeze(0).squeeze(0),
                    hx=current_hx,
                )

                if done:
                    self.episode_returns.append(self.curr_ep_return)
                    self.curr_ep_return = 0.0
                    current_obs_dict, _ = self.env.reset()
                    current_hx = None
                    episode_start = True
                else:
                    current_obs_dict = next_obs_dict
                    current_hx = next_hx
                    episode_start = False

            # Bootstrap value for last state
            input_dict = self._prepare_rollout_input(current_obs_dict)
            _, last_value, _ = self.agent.forward(input_dict, hx=current_hx, dones=None)
            self.buffer.compute_gae(
                last_value=last_value.item(),
                last_done=terminated,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )

            # Global advantage normalization across entire rollout buffer
            adv = self.buffer.advantages
            self.buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        if self.is_auto_exploration:
            self.exploration_scale = float(self._adaptive_scale)

        mean_return = (
            float(np.mean(self.episode_returns)) if len(self.episode_returns) > 0 else 0.0
        )
        return current_obs_dict, current_hx, episode_start, mean_return

    def train_epoch(self) -> Dict[str, float]:
        """Performs PPO mini-batch updates over stored chunks."""
        self.agent.train()
        total_policy_loss = 0.0
        total_val_loss = 0.0
        total_ent_loss = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in self.buffer.get_chunks_generator(self.chunk_batch_size):
                input_dict = {
                    "obs": batch.obs,
                    "prev_action": batch.prev_actions,
                    "prev_reward": batch.prev_rewards,
                }
                if not self.agent.is_visual:
                    input_dict["delta_obs"] = batch.delta_obs

                dist, values, _ = self.agent(input_dict, hx=batch.hx, dones=batch.dones)
                values_sq = values.squeeze(-1)  # [B, T]

                if self.agent.is_discrete:
                    if self.exploration_scale != 1.0:
                        scaled_logits = dist.logits / self.exploration_scale
                        dist_eval = torch.distributions.Categorical(logits=scaled_logits)
                    else:
                        dist_eval = dist
                    new_log_probs = dist_eval.log_prob(batch.actions)  # [B, T]
                    entropy = dist_eval.entropy()                       # [B, T]
                else:
                    if self.exploration_scale != 1.0:
                        scaled_std = dist.scale * self.exploration_scale
                        dist_eval = torch.distributions.Normal(loc=dist.mean, scale=scaled_std)
                    else:
                        dist_eval = dist
                    new_log_probs = dist_eval.log_prob(batch.actions).sum(dim=-1)
                    entropy = dist_eval.entropy().sum(dim=-1)

                # Ratio
                ratio = torch.exp(new_log_probs - batch.old_log_probs)

                # Advantages are already globally normalized across rollout
                norm_adv = batch.advantages

                # Clipped surrogate policy loss
                surr1 = ratio * norm_adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * norm_adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Robust MSE value loss
                val_loss = 0.5 * ((values_sq - batch.returns) ** 2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * val_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()

                # Telemetry: calculate per-block gradient norms
                fe_grads = [p.grad for p in self.agent.front_end.parameters() if p.grad is not None]
                trunk_grads = [p.grad for p in self.agent.trunk.parameters() if p.grad is not None]
                head_grads = [p.grad for p in self.agent.policy_head.parameters() if p.grad is not None]

                grad_norms = {
                    "FrontEnd": float(torch.norm(torch.stack([torch.norm(g) for g in fe_grads])).item()) if fe_grads else 0.0,
                    "Trunk": float(torch.norm(torch.stack([torch.norm(g) for g in trunk_grads])).item()) if trunk_grads else 0.0,
                    "PolicyHead": float(torch.norm(torch.stack([torch.norm(g) for g in head_grads])).item()) if head_grads else 0.0,
                }

                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                clip_frac = float(((ratio - 1.0).abs() > self.clip_range).float().mean().item())
                if self.tracker is not None:
                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    self.tracker.record_training_epoch(
                        policy_loss=policy_loss.item(),
                        value_loss=val_loss.item(),
                        clip_fraction=clip_frac,
                        grad_norms=grad_norms,
                        lr=current_lr,
                    )

                total_policy_loss += policy_loss.item()
                total_val_loss += val_loss.item()
                total_ent_loss += entropy_loss.item()
                n_updates += 1

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_val_loss / max(1, n_updates),
            "entropy_loss": total_ent_loss / max(1, n_updates),
        }

    def train(
        self,
        max_steps: int = 40000,
        target_return: Optional[float] = 475.0,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, float], None]] = None,
        step_callback: Optional[Callable[[], None]] = None,
        verbose: bool = True,
    ) -> float:
        """Trains the agent until max_steps or target_return is reached."""
        obs_dict, _ = self.env.reset(seed=seed)
        hx: Optional[torch.Tensor] = None
        episode_start = True

        total_updates = max(1, max_steps // self.rollout_steps)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_updates,
        )

        best_mean_return = -float("inf")
        iteration = 0

        while self.total_steps < max_steps:
            iteration += 1
            obs_dict, hx, episode_start, mean_return = self.collect_rollouts(
                obs_dict, hx, episode_start, step_callback=step_callback
            )
            metrics = self.train_epoch()

            if len(self.episode_returns) > 0:
                best_mean_return = max(best_mean_return, mean_return)
                if self.tracker is not None:
                    self.tracker.metrics.best_mean_return = best_mean_return

            if verbose and iteration % 2 == 0:
                log_parts = [
                    f"[Step {self.total_steps:6d}/{max_steps}] Mean Return (last 20 ep): {mean_return:.2f}"
                ]
                if self.tracker is not None and self.tracker.metrics.fps > 0:
                    log_parts.append(f"FPS: {self.tracker.metrics.fps:.1f}")
                if self.tracker is not None and len(self.tracker.metrics.confidences) > 0:
                    mean_conf = float(np.mean(self.tracker.metrics.confidences))
                    log_parts.append(f"Confiança: {mean_conf*100:.1f}%")

                calib = float(self._adaptive_scale if self.is_auto_exploration else self.exploration_scale)
                if self.tracker is not None and len(self.tracker.metrics.calibrations) > 0:
                    calib = float(self.tracker.metrics.calibrations[-1])
                log_parts.append(f"Calibração: {calib:.2f}")

                if self.tracker is not None and len(self.tracker.metrics.step_latencies_us) > 0:
                    lat_p50 = float(np.percentile(self.tracker.metrics.step_latencies_us, 50))
                    if lat_p50 >= 1000.0:
                        log_parts.append(f"Latência: {lat_p50/1000.0:.1f}ms")
                    else:
                        log_parts.append(f"Latência: {lat_p50:.0f}µs")

                raw_env = getattr(self.env, "unwrapped", self.env)
                if hasattr(raw_env, "current_depth"):
                    max_d = getattr(raw_env, "max_depth", 5)
                    log_parts.append(f"Profundidade: {raw_env.current_depth}/{max_d}")
                elif hasattr(raw_env, "current_level"):
                    max_l = getattr(raw_env, "max_level", 4)
                    log_parts.append(f"Nível: {raw_env.current_level}/{max_l}")

                print(" | ".join(log_parts))

            if callback is not None:
                callback(self.total_steps, mean_return)

            if (
                target_return is not None
                and len(self.episode_returns) >= 20
                and mean_return >= target_return
            ):
                if verbose:
                    print(
                        f"Target return {target_return} achieved! "
                        f"({mean_return:.2f} at {self.total_steps} steps)"
                    )
                break

        return float(np.mean(self.episode_returns)) if len(self.episode_returns) > 0 else 0.0
