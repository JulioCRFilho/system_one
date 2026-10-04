import io
import threading
import time
from typing import Optional
from PIL import Image, ImageDraw


class VideoFrameBuffer:
    """Buffer thread-safe para armazenamento e distribuição de frames de vídeo MJPEG."""

    def __init__(self, width: int = 480, height: int = 320) -> None:
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame_count = 0
        self._last_update_ts = time.time()
        self._default_frame = self._generate_placeholder("⚡ SYSTEM 1 ENGINE", "STANDBY / AGUARDANDO INÍCIO")
        self._current_frame: bytes = self._default_frame

    def _generate_placeholder(self, title: str, subtitle: str) -> bytes:
        """Gera um frame JPEG estilizado com estética dark mode para quando o jogo estiver ocioso."""
        img = Image.new("RGB", (self.width, self.height), color=(11, 15, 25))
        draw = ImageDraw.Draw(img)
        # Moldura cibernética sutil
        draw.rectangle([6, 6, self.width - 7, self.height - 7], outline=(31, 41, 55), width=2)
        draw.rectangle([12, 12, self.width - 13, self.height - 13], outline=(17, 24, 39), width=1)

        # Textos centralizados aproximadamente
        draw.text((self.width // 2 - 80, self.height // 2 - 20), title, fill=(56, 189, 248))
        draw.text((self.width // 2 - 110, self.height // 2 + 10), subtitle, fill=(148, 163, 184))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def update_frame(self, frame_bytes: bytes) -> None:
        """Atualiza o frame atual e notifica threads de streaming em espera."""
        with self._cond:
            self._current_frame = frame_bytes
            self._frame_count += 1
            self._last_update_ts = time.time()
            self._cond.notify_all()

    def reset_placeholder(self, title: str = "⚡ SYSTEM 1 ENGINE", subtitle: str = "EXECUÇÃO FINALIZADA") -> None:
        """Restaura o frame placeholder com status informativo."""
        self.update_frame(self._generate_placeholder(title, subtitle))

    def get_frame(self, timeout: Optional[float] = None) -> bytes:
        """Retorna o frame mais recente de forma não-bloqueante ou com timeout."""
        with self._lock:
            return self._current_frame

    def wait_for_next_frame(self, timeout: float = 0.5) -> bytes:
        """Aguarda a chegada de um novo frame ou expira retornando o mais recente."""
        with self._cond:
            self._cond.wait(timeout=timeout)
            return self._current_frame

    def wait_for_frame_change(self, last_count: int, timeout: float = 1.0) -> tuple[bytes, int]:
        """Aguarda a publicação de um novo frame ou expira enviando heartbeat do frame atual."""
        with self._cond:
            if self._frame_count == last_count:
                self._cond.wait(timeout=timeout)
            return self._current_frame, self._frame_count
