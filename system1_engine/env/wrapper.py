from typing import Any, Optional, TypedDict, Union
import copy
import gymnasium as gym
import numpy as np
from PIL import Image


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
            # Auto-detect visual observations:
            # 1. Box with 3 dims where min dim in (1, 3, 4) and max spatial dims >= 10
            # 2. Box with 2 dims where both dims >= 10 (grayscale)
            is_visual = False
            if isinstance(env.observation_space, gym.spaces.Box):
                shp = env.observation_space.shape
                if len(shp) == 3 and (shp[0] in (1, 3, 4) or shp[2] in (1, 3, 4)) and max(shp) >= 10:
                    is_visual = True
                elif len(shp) == 2 and min(shp) >= 10:
                    is_visual = True
        self.is_visual = is_visual

        self.visual_channels = 3
        if self.is_visual and isinstance(env.observation_space, gym.spaces.Box):
            shp = env.observation_space.shape
            if len(shp) == 3:
                if shp[0] in (1, 3, 4) and shp[2] not in (1, 3, 4):
                    self.visual_channels = 1 if shp[0] == 1 else 3
                elif shp[2] in (1, 3, 4):
                    self.visual_channels = 1 if shp[2] == 1 else 3
            elif len(shp) == 2:
                self.visual_channels = 1

        self.prev_obs: Optional[np.ndarray] = None
        self.prev_action: Optional[Union[int, np.ndarray]] = None
        self.prev_reward: float = 0.0
        self.latest_obs_dict: Optional[S1Observation] = None

        # Build dictionary observation space
        self._setup_spaces()

    def _get_initial_action(self) -> Union[int, np.ndarray]:
        if isinstance(self.action_space, gym.spaces.Discrete):
            return 0
        elif isinstance(self.action_space, gym.spaces.Box):
            return np.zeros(self.action_space.shape, dtype=np.float32)
        else:
            raise NotImplementedError(f"Unsupported action space: {type(self.action_space)}")

    def _format_visual_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """Padroniza frames visuais para (C, 84, 84) float32 normalizado em [0.0, 1.0]."""
        frame = raw_frame

        # Fast-path para frames já no padrão universal (C, 84, 84) float32 normalizado [0, 1]
        if (
            isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape == (self.visual_channels, 84, 84)
            and frame.dtype == np.float32
            and (frame.size == 0 or (frame.max() <= 1.01 and frame.min() >= -0.01))
        ):
            return np.clip(frame, 0.0, 1.0)

        # Trata dimensões e canais (CHW vs HWC vs HW)
        if frame.ndim == 3:
            if frame.shape[0] in (1, 3, 4) and frame.shape[2] not in (1, 3, 4):
                if frame.shape[0] == 1:
                    frame = frame.squeeze(0)  # [H, W]
                elif frame.shape[0] in (3, 4):
                    frame = np.transpose(frame, (1, 2, 0))  # [H, W, C]
            elif frame.shape[2] == 1:
                frame = frame.squeeze(-1)  # [H, W]

        # Normaliza tipo para uint8 para redimensionamento PIL bilinear de alta fidelidade
        if frame.dtype != np.uint8:
            if frame.max() <= 1.01 and frame.min() >= -0.01:
                frame_uint8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            else:
                frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            frame_uint8 = frame

        if self.visual_channels == 1:
            pil_img = Image.fromarray(frame_uint8)
            if pil_img.mode != "L":
                pil_img = pil_img.convert("L")
            pil_img = pil_img.resize((84, 84), Image.Resampling.BILINEAR)
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0
            return np.expand_dims(arr, axis=0)  # [1, 84, 84]
        else:
            if frame_uint8.ndim == 3 and frame_uint8.shape[2] == 4:
                frame_uint8 = frame_uint8[:, :, :3]
            pil_img = Image.fromarray(frame_uint8)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_img = pil_img.resize((84, 84), Image.Resampling.BILINEAR)
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0
            return np.transpose(arr, (2, 0, 1))  # [3, 84, 84]

    @staticmethod
    def _compute_vector_dim(space: gym.spaces.Space) -> int:
        """Calcula a dimensão total equivalente para espaços de observação vetoriais e discretos."""
        if isinstance(space, gym.spaces.Box):
            return int(np.prod(space.shape))
        elif isinstance(space, gym.spaces.Discrete):
            return int(space.n)
        elif isinstance(space, gym.spaces.MultiBinary):
            if hasattr(space, "shape") and space.shape:
                return int(np.prod(space.shape))
            return int(space.n)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return int(np.sum(space.nvec))
        elif isinstance(space, gym.spaces.Tuple):
            return sum(UniversalS1Wrapper._compute_vector_dim(s) for s in space.spaces)
        else:
            raise NotImplementedError(
                f"Espaço de observação não suportado: {type(space)}. "
                "Espaços suportados: Box, Discrete, MultiDiscrete, MultiBinary, Tuple."
            )

    @classmethod
    def _vectorize_obs(cls, raw_obs: Any, space: gym.spaces.Space) -> np.ndarray:
        """Converte qualquer observação não-visual em um vetor float32 1D padronizado."""
        if isinstance(space, gym.spaces.Box):
            return np.asarray(raw_obs, dtype=np.float32).flatten()
        elif isinstance(space, gym.spaces.Discrete):
            n = int(space.n)
            vec = np.zeros(n, dtype=np.float32)
            idx = int(raw_obs)
            if 0 <= idx < n:
                vec[idx] = 1.0
            return vec
        elif isinstance(space, gym.spaces.MultiBinary):
            return np.asarray(raw_obs, dtype=np.float32).flatten()
        elif isinstance(space, gym.spaces.MultiDiscrete):
            total_dim = int(np.sum(space.nvec))
            vec = np.zeros(total_dim, dtype=np.float32)
            offset = 0
            for val, n in zip(raw_obs, space.nvec):
                idx = int(val)
                if 0 <= idx < n:
                    vec[offset + idx] = 1.0
                offset += int(n)
            return vec
        elif isinstance(space, gym.spaces.Tuple):
            sub_vecs = [cls._vectorize_obs(sub_val, sub_sp) for sub_val, sub_sp in zip(raw_obs, space.spaces)]
            return np.concatenate(sub_vecs, axis=0).astype(np.float32)
        else:
            return np.asarray(raw_obs, dtype=np.float32).flatten()

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
            # Visual observations are standardized to (C, 84, 84) and stacked to (2 * C, 84, 84)
            spaces["obs"] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2 * self.visual_channels, 84, 84),
                dtype=np.float32,
            )
        else:
            obs_dim = self._compute_vector_dim(obs_space)
            if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 1:
                low = obs_space.low.astype(np.float32)
                high = obs_space.high.astype(np.float32)
            elif isinstance(obs_space, gym.spaces.Box):
                low = obs_space.low.flatten().astype(np.float32)
                high = obs_space.high.flatten().astype(np.float32)
            elif isinstance(obs_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)):
                low = np.zeros(obs_dim, dtype=np.float32)
                high = np.ones(obs_dim, dtype=np.float32)
            else:
                low = np.full((obs_dim,), -np.inf, dtype=np.float32)
                high = np.full((obs_dim,), np.inf, dtype=np.float32)

            spaces["obs"] = gym.spaces.Box(
                low=low,
                high=high,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            spaces["delta_obs"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
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

        if self.is_visual:
            current_obs = self._format_visual_frame(np.asarray(raw_obs))
            self.prev_obs = np.array(current_obs, copy=True)
            self.prev_action = self._get_initial_action()
            self.prev_reward = 0.0

            stacked_obs = np.concatenate([current_obs, current_obs], axis=0)
            obs_dict: S1Observation = {
                "obs": stacked_obs,
                "delta_obs": None,
                "prev_action": copy.deepcopy(self.prev_action),
                "prev_reward": float(self.prev_reward),
            }
        else:
            current_obs = self._vectorize_obs(raw_obs, self.env.observation_space)
            self.prev_obs = np.array(current_obs, copy=True)
            self.prev_action = self._get_initial_action()
            self.prev_reward = 0.0

            delta_obs = np.zeros_like(current_obs, dtype=np.float32)
            obs_dict = {
                "obs": current_obs,
                "delta_obs": delta_obs,
                "prev_action": copy.deepcopy(self.prev_action),
                "prev_reward": float(self.prev_reward),
            }

        self.latest_obs_dict = obs_dict
        return obs_dict, info

    def step(
        self, action: Any
    ) -> tuple[S1Observation, float, bool, bool, dict[str, Any]]:
        """Takes an environment step and computes temporal differential state."""
        assert self.prev_obs is not None, "step() called before reset()"
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        reward_float = float(reward)

        if self.is_visual:
            current_obs = self._format_visual_frame(np.asarray(next_obs))
            stacked_obs = np.concatenate([current_obs, self.prev_obs], axis=0)
            obs_dict: S1Observation = {
                "obs": stacked_obs,
                "delta_obs": None,
                "prev_action": copy.deepcopy(action),
                "prev_reward": reward_float,
            }
        else:
            current_obs = self._vectorize_obs(next_obs, self.env.observation_space)
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
        self.latest_obs_dict = obs_dict

        return obs_dict, reward_float, terminated, truncated, info

