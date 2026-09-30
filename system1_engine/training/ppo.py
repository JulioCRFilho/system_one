from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper
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

        # Persistent episode statistics across rollout batches
        self.episode_returns: deque[float] = deque(maxlen=20)
        self.curr_ep_return: float = 0.0
        self.total_steps: int = 0

    def collect_rollouts(
        self,
        current_obs_dict: Dict[str, Any],
        current_hx: Optional[torch.Tensor],
        episode_start: bool,
    ) -> Tuple[Dict[str, Any], Optional[torch.Tensor], bool, float]:
        """Collects rollout_steps interactions in the environment."""
        self.buffer.reset()
        self.agent.eval()

        with torch.no_grad():
            for _ in range(self.rollout_steps):
                # Prepare tensor inputs for action selection
                obs_t = (
                    torch.from_numpy(current_obs_dict["obs"])
                    .float()
                    .unsqueeze(0)
                    .to(self.device)
                )
                prev_act_raw = current_obs_dict["prev_action"]
                if self.agent.is_discrete:
                    act_t = torch.tensor([[int(prev_act_raw)]], dtype=torch.long, device=self.device)
                else:
                    act_t = (
                        torch.from_numpy(np.asarray(prev_act_raw, dtype=np.float32))
                        .unsqueeze(0)
                        .to(self.device)
                    )

                rew_t = (
                    torch.tensor([[[float(current_obs_dict["prev_reward"])]]], dtype=torch.float32)
                    .to(self.device)
                )

                input_dict = {
                    "obs": obs_t,
                    "prev_action": act_t,
                    "prev_reward": rew_t,
                }
                if not self.agent.is_visual:
                    delta_t = (
                        torch.from_numpy(current_obs_dict["delta_obs"])
                        .float()
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    input_dict["delta_obs"] = delta_t

                # Step agent
                action, log_prob, value, next_hx = self.agent.get_action(
                    input_dict, hx=current_hx, deterministic=False
                )

                # Environment step
                if self.agent.is_discrete:
                    env_action = int(action.item())
                else:
                    env_action = action.squeeze(0).squeeze(0).cpu().numpy()

                next_obs_dict, reward, terminated, truncated, _ = self.env.step(env_action)
                done = terminated or truncated

                self.curr_ep_return += reward
                self.total_steps += 1

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
            obs_t = (
                torch.from_numpy(current_obs_dict["obs"])
                .float()
                .unsqueeze(0)
                .to(self.device)
            )
            prev_act_raw = current_obs_dict["prev_action"]
            if self.agent.is_discrete:
                act_t = torch.tensor([[int(prev_act_raw)]], dtype=torch.long, device=self.device)
            else:
                act_t = (
                    torch.from_numpy(np.asarray(prev_act_raw, dtype=np.float32))
                    .unsqueeze(0)
                    .to(self.device)
                )

            rew_t = (
                torch.tensor([[[float(current_obs_dict["prev_reward"])]]], dtype=torch.float32)
                .to(self.device)
            )

            input_dict = {
                "obs": obs_t,
                "prev_action": act_t,
                "prev_reward": rew_t,
            }
            if not self.agent.is_visual:
                delta_t = (
                    torch.from_numpy(current_obs_dict["delta_obs"])
                    .float()
                    .unsqueeze(0)
                    .to(self.device)
                )
                input_dict["delta_obs"] = delta_t

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
                    new_log_probs = dist.log_prob(batch.actions)  # [B, T]
                    entropy = dist.entropy()                       # [B, T]
                else:
                    new_log_probs = dist.log_prob(batch.actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1)

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
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

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
                obs_dict, hx, episode_start
            )
            metrics = self.train_epoch()

            if len(self.episode_returns) >= 20:
                best_mean_return = max(best_mean_return, mean_return)

            if verbose and iteration % 2 == 0:
                print(
                    f"[Step {self.total_steps:6d}/{max_steps}] "
                    f"Mean Return (last 20 ep): {mean_return:.2f} | "
                    f"Best: {best_mean_return:.2f} | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f}"
                )

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
