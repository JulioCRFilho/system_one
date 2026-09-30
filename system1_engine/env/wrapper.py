from typing import Any, Optional, TypedDict, Union
import copy
import gymnasium as gym
import numpy as np


class S1Observation(TypedDict):
    obs: np.ndarray
    delta_obs: Optional[np.ndarray]
    prev_action: Union[int, np.ndarray]
    prev_reward: float


class UniversalS1Wrapper(gym.Wrapper):
    """Universal System 1 Environment Wrapper.

    Augments standard observations with:
    - delta_obs: s_t - s_{t-1} for vector observations, or channel stacking for visual.
    - prev_action: a_{t-1}.
    - prev_reward: r_{t-1}.

    Prevents causal blindness and cyclic attractors in reflexive policies.
    """

    def __init__(self, env: gym.Env, is_visual: Optional[bool] = None) -> None:
        super().__init__(env)
        if is_visual is None:
            # Auto-detect visual observations: Box with ndim == 3
            is_visual = (
                isinstance(env.observation_space, gym.spaces.Box)
                and len(env.observation_space.shape) == 3
            )
        self.is_visual = is_visual

        self.prev_obs: Optional[np.ndarray] = None
        self.prev_action: Optional[Union[int, np.ndarray]] = None
        self.prev_reward: float = 0.0

        # Build dictionary observation space
        self._setup_spaces()

    def _get_initial_action(self) -> Union[int, np.ndarray]:
        if isinstance(self.action_space, gym.spaces.Discrete):
            return 0
        elif isinstance(self.action_space, gym.spaces.Box):
            return np.zeros(self.action_space.shape, dtype=np.float32)
        else:
            raise NotImplementedError(f"Unsupported action space: {type(self.action_space)}")

    def _setup_spaces(self) -> None:
        obs_space = self.env.observation_space
        act_space = self.action_space

        spaces: dict[str, gym.spaces.Space] = {
            "prev_action": act_space,
            "prev_reward": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
        }

        if self.is_visual:
            assert isinstance(obs_space, gym.spaces.Box), "Visual envs must have Box observation space"
            # Visual observations are stacked along channel dimension (2 * C, H, W)
            c, h, w = obs_space.shape
            spaces["obs"] = gym.spaces.Box(
                low=float(obs_space.low.min()),
                high=float(obs_space.high.max()),
                shape=(2 * c, h, w),
                dtype=np.float32,
            )
        else:
            assert isinstance(obs_space, gym.spaces.Box), "Vector envs must have Box observation space"
            spaces["obs"] = gym.spaces.Box(
                low=obs_space.low.astype(np.float32),
                high=obs_space.high.astype(np.float32),
                shape=obs_space.shape,
                dtype=np.float32,
            )
            spaces["delta_obs"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_space.shape,
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(spaces)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[S1Observation, dict[str, Any]]:
        """Resets the environment and flushes all temporal and causal memory buffers."""
        raw_obs, info = self.env.reset(seed=seed, options=options)
        current_obs = np.array(raw_obs, dtype=np.float32)

        # Strictly flush buffers - no cross-episode state leakage
        self.prev_obs = np.array(current_obs, copy=True)
        self.prev_action = self._get_initial_action()
        self.prev_reward = 0.0

        if self.is_visual:
            # For vision at t=0: stack with zeros or copy of current frame
            # Concatenating [obs, obs] along channel axis (2*C, H, W)
            stacked_obs = np.concatenate([current_obs, current_obs], axis=0)
            obs_dict: S1Observation = {
                "obs": stacked_obs,
                "delta_obs": None,
                "prev_action": copy.deepcopy(self.prev_action),
                "prev_reward": float(self.prev_reward),
            }
        else:
            delta_obs = np.zeros_like(current_obs, dtype=np.float32)
            obs_dict = {
                "obs": current_obs,
                "delta_obs": delta_obs,
                "prev_action": copy.deepcopy(self.prev_action),
                "prev_reward": float(self.prev_reward),
            }

        return obs_dict, info

    def step(
        self, action: Any
    ) -> tuple[S1Observation, float, bool, bool, dict[str, Any]]:
        """Takes an environment step and computes temporal differential state."""
        assert self.prev_obs is not None, "step() called before reset()"
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        current_obs = np.array(next_obs, dtype=np.float32)

        reward_float = float(reward)

        if self.is_visual:
            stacked_obs = np.concatenate([current_obs, self.prev_obs], axis=0)
            obs_dict: S1Observation = {
                "obs": stacked_obs,
                "delta_obs": None,
                "prev_action": copy.deepcopy(action),
                "prev_reward": reward_float,
            }
        else:
            delta_obs = current_obs - self.prev_obs
            obs_dict = {
                "obs": current_obs,
                "delta_obs": delta_obs,
                "prev_action": copy.deepcopy(action),
                "prev_reward": reward_float,
            }

        # Update buffers for next step
        self.prev_obs = np.array(current_obs, copy=True)
        self.prev_action = copy.deepcopy(action)
        self.prev_reward = reward_float

        return obs_dict, reward_float, terminated, truncated, info
