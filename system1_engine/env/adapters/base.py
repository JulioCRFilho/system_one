from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import time
import gymnasium as gym
import numpy as np


class BaseGameAdapter(gym.Env, ABC):
    """Classe base abstrata para adaptadores de integração do System 1.

    Todos os 3 níveis (WindowCapture, MemoryHook, NativeEngine) herdam desta interface,
    garantindo compatibilidade estrita com gym.Env e com o UniversalS1Wrapper.

    Responsabilidades:
      - Padronização do contrato gym.Env (reset, step, close).
      - Rastreamento de telemetria diagnóstica (FPS real, latência de captura/step, contagem).
      - Suporte nativo tanto a observações vetoriais quanto visuais (C, H, W).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        is_visual: bool = False,
        target_fps: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.is_visual = is_visual
        self.target_fps = target_fps

        # Telemetria e diagnóstico
        self.step_count: int = 0
        self.episode_count: int = 0
        self.total_episode_reward: float = 0.0
        self.last_step_time_ms: float = 0.0
        self._last_timestamp: float = 0.0
        self._fps_window: list[float] = []

    @property
    def effective_fps(self) -> float:
        """Calcula a taxa efetiva de FPS com base nos últimos passos."""
        if len(self._fps_window) < 2:
            return float(self.target_fps or 0.0)
        dt_total = self._fps_window[-1] - self._fps_window[0]
        if dt_total <= 0.0:
            return 0.0
        return (len(self._fps_window) - 1) / dt_total

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reinicia o ambiente e buffers de telemetria."""
        super().reset(seed=seed)
        self.episode_count += 1
        self.step_count = 0
        self.total_episode_reward = 0.0
        self._fps_window.clear()
        self._last_timestamp = time.perf_counter()

        obs, info = self._reset_impl(seed=seed, options=options)
        info = info or {}
        info["adapter_type"] = self.__class__.__name__
        info["episode"] = self.episode_count
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Executa um passo físico/lógico e registra telemetria."""
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = self._step_impl(action)
        t1 = time.perf_counter()

        # Telemetria
        self.last_step_time_ms = (t1 - t0) * 1000.0
        self.step_count += 1
        self.total_episode_reward += float(reward)

        self._fps_window.append(t1)
        if len(self._fps_window) > 30:
            self._fps_window.pop(0)

        info = info or {}
        info["step"] = self.step_count
        info["step_time_ms"] = self.last_step_time_ms
        info["effective_fps"] = self.effective_fps
        info["total_episode_reward"] = self.total_episode_reward

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Optional[np.ndarray]:
        """Retorna frame visual para streaming in-browser ou renderização."""
        return self._render_impl()

    def _render_impl(self) -> Optional[np.ndarray]:
        return None

    def close(self) -> None:
        """Encerra recursos de hardware, sockets ou processos anexados."""
        self._close_impl()
        super().close()

    # Métodos abstratos específicos de cada nível
    @abstractmethod
    def _reset_impl(
        self, seed: Optional[int], options: Optional[Dict[str, Any]]
    ) -> Tuple[Any, Dict[str, Any]]:
        pass

    @abstractmethod
    def _step_impl(
        self, action: Any
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        pass

    @abstractmethod
    def _close_impl(self) -> None:
        pass
