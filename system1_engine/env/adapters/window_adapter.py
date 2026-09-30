from typing import Any, Callable, Dict, Optional, Tuple, Union
import time
import gymnasium as gym
import numpy as np
from PIL import Image

from system1_engine.env.adapters.base import BaseGameAdapter

# Importação defensiva de bibliotecas de sistema operacional
try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    from pynput.keyboard import Controller as KeyboardController, Key
    HAS_PYNPUT = True
except (ImportError, Exception):
    HAS_PYNPUT = False
    KeyboardController = None  # type: ignore
    Key = None  # type: ignore


class WindowCaptureEnv(BaseGameAdapter):
    """Nível 1: Adaptador de Caixa-Preta Total (Screen Capture & OS Input).

    Opera sobre qualquer janela de jogo ou aplicativo em tempo real:
      - Captura contínua de frames via mss (GPU/OS buffer) redimensionados para (C, 84, 84).
      - Emulação de periféricos (teclado/mouse) via pynput.
      - Frame pacing assíncrono com cadência física controlada (ex.: 30 FPS).
      - Função de recompensa e término plugáveis via visão computacional.
    """

    def __init__(
        self,
        window_bbox: Optional[Dict[str, int]] = None,
        actions_map: Optional[Dict[int, Union[str, None]]] = None,
        target_fps: int = 30,
        grayscale: bool = True,
        reward_fn: Optional[Callable[[np.ndarray, Optional[np.ndarray], Dict[str, Any]], float]] = None,
        done_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], bool]] = None,
        reset_action: Optional[Union[str, Callable[[], None]]] = None,
        mock_capture_source: Optional[Callable[[], np.ndarray]] = None,
        key_press_duration: float = 0.02,
    ) -> None:
        self.channels = 1 if grayscale else 3
        self.grayscale = grayscale
        self.target_fps = target_fps
        self.frame_interval = 1.0 / max(1, target_fps)
        self.key_press_duration = key_press_duration

        # Bounding box no formato mss: {"top": int, "left": int, "width": int, "height": int}
        self.window_bbox = window_bbox or {"top": 0, "left": 0, "width": 800, "height": 600}
        self.actions_map = actions_map or {0: None}  # Padrão: 0 = No-Op

        self.reward_fn = reward_fn
        self.done_fn = done_fn
        self.reset_action = reset_action
        self.mock_capture_source = mock_capture_source

        # Espaços Gymnasium padronizados
        observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.channels, 84, 84),
            dtype=np.float32,
        )
        action_space = gym.spaces.Discrete(len(self.actions_map))

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            is_visual=True,
            target_fps=float(target_fps),
        )

        # Inicialização do MSS
        mss_cls = getattr(mss, "MSS", getattr(mss, "mss", None)) if HAS_MSS else None
        self._sct = mss_cls() if (mss_cls is not None and mock_capture_source is None) else None

        # Inicialização do Teclado
        self._keyboard = None
        if HAS_PYNPUT and KeyboardController is not None:
            try:
                self._keyboard = KeyboardController()
            except Exception:
                self._keyboard = None

        self._prev_frame: Optional[np.ndarray] = None
        self._last_step_end: float = time.perf_counter()

    def _capture_raw_frame(self) -> np.ndarray:
        """Captura frame bruto da tela ou de fonte mock."""
        if self.mock_capture_source is not None:
            return self.mock_capture_source()

        if self._sct is not None:
            sct_img = self._sct.grab(self.window_bbox)
            # mss retorna BGRA
            arr = np.asarray(sct_img, dtype=np.uint8)
            # Converte BGRA -> RGB
            return arr[:, :, [2, 1, 0]]

        # Fallback de segurança se mss não estiver acessível
        return np.zeros(
            (self.window_bbox.get("height", 600), self.window_bbox.get("width", 800), 3),
            dtype=np.uint8,
        )

    def _preprocess_frame(self, raw_img: np.ndarray) -> np.ndarray:
        """Redimensiona e formata para (C, 84, 84) normalizado em [0.0, 1.0]."""
        pil_img = Image.fromarray(raw_img)
        if self.grayscale:
            pil_img = pil_img.convert("L").resize((84, 84), Image.Resampling.BILINEAR)
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0
            return np.expand_dims(arr, axis=0)  # [1, 84, 84]
        else:
            pil_img = pil_img.convert("RGB").resize((84, 84), Image.Resampling.BILINEAR)
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0
            # Transpõe [84, 84, 3] -> [3, 84, 84]
            return np.transpose(arr, (2, 0, 1))

    def _dispatch_action(self, action: int) -> None:
        """Emula entrada de hardware (teclado)."""
        key_target = self.actions_map.get(action)
        if key_target is None or self._keyboard is None:
            return

        try:
            # Verifica se é uma tecla especial nomeada (ex.: 'space', 'enter')
            if hasattr(Key, key_target):
                k = getattr(Key, key_target)
            else:
                k = key_target

            self._keyboard.press(k)
            if self.key_press_duration > 0:
                time.sleep(self.key_press_duration)
            self._keyboard.release(k)
        except Exception:
            # Protege o loop em caso de erro no driver de entrada
            pass

    def _reset_impl(
        self, seed: Optional[int], options: Optional[Dict[str, Any]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Executa ação de reset caso configurada (ex: pressionar tecla 'r')
        if self.reset_action is not None:
            if callable(self.reset_action):
                self.reset_action()
            elif isinstance(self.reset_action, str) and self._keyboard is not None:
                try:
                    k = getattr(Key, self.reset_action) if hasattr(Key, self.reset_action) else self.reset_action
                    self._keyboard.press(k)
                    time.sleep(0.05)
                    self._keyboard.release(k)
                except Exception:
                    pass

        raw_frame = self._capture_raw_frame()
        obs = self._preprocess_frame(raw_frame)
        self._prev_frame = obs.copy()
        self._last_step_end = time.perf_counter()

        return obs, {"window_bbox": self.window_bbox}

    def _step_impl(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 1. Envia ação para o sistema operacional
        self._dispatch_action(action)

        # 2. Frame Pacing (Cadência física assíncrona)
        now = time.perf_counter()
        elapsed = now - self._last_step_end
        sleep_needed = self.frame_interval - elapsed
        if sleep_needed > 0:
            time.sleep(sleep_needed)

        # 3. Captura novo frame
        raw_frame = self._capture_raw_frame()
        obs = self._preprocess_frame(raw_frame)

        info: Dict[str, Any] = {"raw_shape": raw_frame.shape}

        # 4. Cálculo de Recompensa
        reward = 0.0
        if self.reward_fn is not None:
            reward = float(self.reward_fn(obs, self._prev_frame, info))

        # 5. Critério de Término
        terminated = False
        if self.done_fn is not None:
            terminated = bool(self.done_fn(obs, info))

        truncated = False
        self._prev_frame = obs.copy()
        self._last_step_end = time.perf_counter()

        return obs, reward, terminated, truncated, info

    def _close_impl(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
