import glob
import http.server
import json
import os
import socketserver
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from system1_engine.hud.frame_buffer import VideoFrameBuffer
from system1_engine.hud.runner import HUDProcessRunner
from system1_engine.transfer.manager import KnowledgeTransferManager


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Servidor TCP multi-thread com liberação rápida de portas e supressão de erros de desconexão."""

    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        exc_type, _, _ = sys.exc_info()
        if exc_type in (
            ConnectionResetError,
            BrokenPipeError,
            ConnectionAbortedError,
            TimeoutError,
        ):
            return
        super().handle_error(request, client_address)


class HUDServer:
    """Servidor Web do HUD Operacional do System 1 Engine."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8050,
        open_browser: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self.runner = HUDProcessRunner()
        self.frame_buffer = VideoFrameBuffer()
        self._lock = threading.Lock()
        self.latest_telemetry: Dict[str, Any] = {
            "fps": 0.0,
            "total_steps": 0,
            "mean_return_20": 0.0,
            "best_return": 0.0,
            "best_mean_return": 0.0,
            "latency_p50_us": 0.0,
            "latency_p99_us": 0.0,
            "mean_confidence": 0.0,
            "mean_uncertainty": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "grad_norms": {"FrontEnd": 0.0, "Trunk": 0.0, "PolicyHead": 0.0},
        }

        self.server: Optional[ThreadedTCPServer] = None
        self.thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

        # Caminho do template HTML do dashboard
        self.html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")

    def get_dashboard_html(self) -> str:
        """Lê o template HTML do disco dinamicamente para permitir recarregamento imediato (Cmd+R)."""
        if os.path.exists(self.html_path):
            try:
                with open(self.html_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
        return "<h1>Dashboard template not found</h1>"

    def scan_checkpoints(self) -> List[str]:
        """Varre o diretório do projeto em busca de modelos e checkpoints .pt disponíveis."""
        candidates = glob.glob("*.pt") + glob.glob("**/*.pt", recursive=True)
        valid = set()
        for p in candidates:
            # Ignora pastas virtuais, git e caches
            if any(part.startswith(".") or part == ".venv" or part == "build" for part in p.split(os.sep)):
                continue
            valid.add(p)
        return sorted(list(valid))

    def scan_checkpoints_detailed(self) -> List[Dict[str, Any]]:
        """Retorna detalhes enriquecidos (tamanho em MB, data, ambiente de origem e tensores) de cada checkpoint."""
        candidates = self.scan_checkpoints()
        results = []
        for path in candidates:
            try:
                stat = os.stat(path)
                size_mb = stat.st_size / (1024 * 1024)
                mtime_str = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(stat.st_mtime))
                meta = KnowledgeTransferManager.inspect_checkpoint(path)
                results.append({
                    "path": path,
                    "filename": os.path.basename(path),
                    "size_mb": round(size_mb, 2),
                    "mtime": mtime_str,
                    "env_id": meta.get("env_id"),
                    "obs_dim": meta.get("obs_dim"),
                    "act_dim": meta.get("act_dim"),
                    "final_return": meta.get("final_return"),
                    "steps": meta.get("steps"),
                })
            except Exception:
                results.append({"path": path, "filename": os.path.basename(path), "size_mb": 0.0, "mtime": "—"})
        return results

    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        """Detecta aceleradores de hardware disponíveis (Apple Silicon MPS ou NVIDIA CUDA)."""
        mps_available = False
        cuda_available = False
        device_name = "CPU Universal"
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if cuda_available:
                name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA"
                device_name = f"NVIDIA CUDA ({name})"
            elif mps_available:
                device_name = "Apple Silicon GPU (MPS)"
            else:
                device_name = "CPU Universal"
        except Exception:
            pass
        return {
            "mps_available": mps_available,
            "cuda_available": cuda_available,
            "gpu_available": mps_available or cuda_available,
            "device_name": device_name,
        }

    def update_telemetry(self, data: Dict[str, Any]) -> None:
        with self._lock:
            self.latest_telemetry.update(data)

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.latest_telemetry)

    def start(self) -> None:
        if self._running:
            return

        server_instance = self
        runner = self.runner
        frame_buffer = self.frame_buffer
        stop_event = self._stop_event

        class HUDRequestHandler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def handle(self) -> None:
                try:
                    super().handle()
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, TimeoutError, OSError):
                    pass

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    encoded_html = server_instance.get_dashboard_html().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(encoded_html)))
                    self.end_headers()
                    self.wfile.write(encoded_html)

                elif self.path == "/video_feed":
                    # MJPEG Video Streaming Endpoint
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()

                    try:
                        while not stop_event.is_set():
                            frame = frame_buffer.wait_for_next_frame(timeout=0.1)
                            header = (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                b"Content-Length: " + str(len(frame)).encode("ascii") + b"\r\n\r\n"
                            )
                            self.wfile.write(header + frame + b"\r\n")
                            self.wfile.flush()
                            time.sleep(0.02)  # ~50 FPS máx para o streaming HTTP
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass

                elif self.path == "/stream":
                    # SSE Real-time Telemetry & Log Stream
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    client_log_cursor = 0
                    try:
                        while not stop_event.is_set():
                            new_logs, client_log_cursor = runner.get_logs(client_log_cursor)
                            snap = server_instance.get_telemetry_snapshot()
                            runner_state = runner.get_state()

                            payload = {
                                "telemetry": snap,
                                "runner": runner_state,
                                "new_logs": new_logs,
                            }

                            data_line = f"data: {json.dumps(payload)}\n\n"
                            self.wfile.write(data_line.encode("utf-8"))
                            self.wfile.flush()
                            time.sleep(0.08)  # ~12 Hz
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass

                elif self.path == "/api/state":
                    resp = {
                        "runner": runner.get_state(),
                        "hardware": server_instance.get_hardware_info(),
                        "checkpoints": server_instance.scan_checkpoints(),
                        "checkpoints_detailed": server_instance.scan_checkpoints_detailed(),
                        "telemetry": server_instance.get_telemetry_snapshot(),
                    }
                    encoded_resp = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded_resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(encoded_resp)

                elif self.path == "/api/checkpoints":
                    resp = {
                        "checkpoints": server_instance.scan_checkpoints(),
                        "checkpoints_detailed": server_instance.scan_checkpoints_detailed(),
                    }
                    encoded_resp = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded_resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(encoded_resp)

                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

            def do_POST(self) -> None:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)

                if self.path == "/api/action":
                    try:
                        data = json.loads(body.decode("utf-8"))
                        action = data.get("action")
                        if action == "start":
                            cfg = data.get("config", {})
                            url = f"http://{server_instance.host}:{server_instance.port}"
                            success, msg = runner.start(cfg, url)
                            resp = {"success": success, "message": msg}
                        elif action == "stop":
                            success, msg = runner.stop()
                            frame_buffer.reset_placeholder("⚡ SYSTEM 1 ENGINE", "TAREFA INTERROMPIDA PELO USUÁRIO")
                            resp = {"success": success, "message": msg}
                        else:
                            resp = {"success": False, "message": f"Ação desconhecida: {action}"}
                    except Exception as err:
                        resp = {"success": False, "message": str(err)}

                    encoded_action = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded_action)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(encoded_action)

                elif self.path == "/api/internal/telemetry":
                    try:
                        data = json.loads(body.decode("utf-8"))
                        server_instance.update_telemetry(data)
                        self.send_response(200)
                    except Exception:
                        self.send_response(400)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

                elif self.path == "/api/internal/frame":
                    frame_buffer.update_frame(body)
                    self.send_response(200)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

            def log_message(self, format: str, *args) -> None:
                # Silencia logs verbosos no console
                return

        self._stop_event.clear()
        self.server = ThreadedTCPServer((self.host, self.port), HUDRequestHandler)
        self._running = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        dashboard_url = f"http://{self.host}:{self.port}"
        print("\n" + "=" * 70)
        print("🎮 SYSTEM 1 ENGINE — VISUAL CONTROL HUD INICIALIZADO")
        print(f"🌐 Acesse no seu navegador em: {dashboard_url}")
        print("=" * 70)

        if self.open_browser:
            def _open() -> None:
                time.sleep(0.2)
                try:
                    import webbrowser
                    webbrowser.open(dashboard_url)
                except Exception:
                    pass

            threading.Thread(target=_open, daemon=True).start()

    def stop(self) -> None:
        if self._running:
            self._stop_event.set()
            self._running = False
            self.runner.stop()
            if self.server is not None:
                self.server.shutdown()
                self.server.server_close()
                self.server = None

    def __enter__(self) -> "HUDServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
