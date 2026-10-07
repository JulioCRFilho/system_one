import collections
import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class HUDProcessRunner:
    """Gerenciador de subprocesso para treinos e avaliações disparados pelo HUD."""

    def __init__(self, max_log_history: int = 500) -> None:
        self.max_log_history = max_log_history
        self._lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
        self.status: str = "IDLE"  # IDLE, RUNNING, STOPPED, COMPLETED, ERROR
        self.current_config: Dict[str, Any] = {}
        self.logs: collections.deque[str] = collections.deque(maxlen=max_log_history)
        self.log_counter = 0
        self._reader_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_requested = False

    def is_running(self) -> bool:
        with self._lock:
            return self.process is not None and self.process.poll() is None

    def start(self, config: Dict[str, Any], server_url: str) -> Tuple[bool, str]:
        """Inicia um novo processo filho de execução em segundo plano."""
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                return False, "Já existe uma execução em andamento. Interrompa-a antes de iniciar outra."

            self._stop_requested = False
            self.current_config = config
            self.status = "RUNNING"
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] ▶️ Iniciando tarefa ({config.get('mode', 'train')})...")
            self.log_counter += 1

            cmd = [
                sys.executable,
                "-u",  # Unbuffered stdout/stderr
                "-m",
                "system1_engine.hud.worker",
                "--config",
                json.dumps(config),
                "--server-url",
                server_url,
            ]

            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=os.environ.copy(),
                )
            except Exception as e:
                self.status = "ERROR"
                err_msg = f"Falha ao iniciar subprocesso: {e}"
                self.logs.append(f"[{time.strftime('%H:%M:%S')}] ❌ {err_msg}")
                self.log_counter += 1
                return False, err_msg

        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()

        self._monitor_thread = threading.Thread(target=self._monitor_process, daemon=True)
        self._monitor_thread.start()

        return True, "Execução iniciada com sucesso."

    def stop(self) -> Tuple[bool, str]:
        """Interrompe a execução atual com encerramento gracioso via SIGINT / SIGTERM."""
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                self.status = "IDLE"
                return True, "Nenhuma execução ativa para interromper."

            self._stop_requested = True
            proc = self.process

        # Tenta encerrar graciosamente via SIGINT (Ctrl+C simulado)
        try:
            proc.send_signal(signal.SIGINT)
        except OSError:
            pass

        # Aguarda até 2.0s pelo término limpo
        t0 = time.time()
        while time.time() - t0 < 2.0:
            if proc.poll() is not None:
                break
            time.sleep(0.05)

        # Força SIGTERM caso ainda esteja ativo
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
            time.sleep(0.2)

        # Último recurso: SIGKILL
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

        with self._lock:
            self.status = "STOPPED"
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] ⏹️ Execução interrompida pelo usuário.")
            self.log_counter += 1

        return True, "Execução interrompida."

    def _read_stdout(self) -> None:
        """Lê linhas de stdout do processo filho e alimenta o histórico de logs."""
        proc = self.process
        if proc is None or proc.stdout is None:
            return

        for line in iter(proc.stdout.readline, ""):
            line_clean = line.rstrip()
            if line_clean:
                with self._lock:
                    self.logs.append(line_clean)
                    self.log_counter += 1
        proc.stdout.close()

    def _monitor_process(self) -> None:
        """Monitora o encerramento do processo e atualiza o estado correspondente."""
        proc = self.process
        if proc is None:
            return

        exit_code = proc.wait()

        with self._lock:
            if self._stop_requested:
                self.status = "STOPPED"
            elif exit_code == 0:
                self.status = "COMPLETED"
                self.logs.append(f"[{time.strftime('%H:%M:%S')}] 🏁 Tarefa concluída com sucesso (código 0).")
                self.log_counter += 1
            else:
                self.status = "ERROR"
                self.logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠️ Processo finalizado com erro (código {exit_code}).")
                self.log_counter += 1

    def clear_logs(self) -> None:
        """Limpa o buffer histórico de logs acumulados."""
        with self._lock:
            self.logs.clear()
            self.log_counter = 0

    def get_logs(self, since_index: int = 0) -> Tuple[List[str], int]:
        """Retorna as linhas de log acumuladas desde um índice específico."""
        with self._lock:
            all_logs = list(self.logs)
            total = self.log_counter
            # Se o cliente perdeu muitos logs por rotação, envia os disponíveis
            start_offset = max(0, total - len(all_logs))
            slice_start = max(0, since_index - start_offset)
            return all_logs[slice_start:], total

    def get_state(self) -> Dict[str, Any]:
        """Retorna o estado operacional consolidado do executor."""
        with self._lock:
            return {
                "status": self.status,
                "is_running": self.process is not None and self.process.poll() is None,
                "pid": self.process.pid if self.process is not None else None,
                "config": self.current_config,
                "total_logs": self.log_counter,
            }
