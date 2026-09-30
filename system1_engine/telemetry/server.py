import http.server
import json
import socketserver
import threading
import time
from typing import Optional

from system1_engine.telemetry.tracker import LiveStatsTracker

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>System 1 Engine — Telemetry Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { background-color: #0b0f19; color: #e2e8f0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    .card { background-color: #111827; border: 1px solid #1f2937; border-radius: 0.5rem; }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  </style>
</head>
<body class="p-6">
  <!-- Top Bar -->
  <header class="flex flex-wrap justify-between items-center mb-6 pb-4 border-b border-gray-800 gap-4">
    <div>
      <div class="flex items-center gap-3">
        <span class="text-2xl font-black text-cyan-400">⚡ SYSTEM 1 ENGINE</span>
        <span class="px-2 py-0.5 rounded text-xs bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono">LIVE TELEMETRY</span>
      </div>
      <p class="text-xs text-gray-500 mt-1">Amortized Reflex Core & Pure PyTorch Recurrent RL Telemetry Bus</p>
    </div>
    <div class="flex flex-wrap gap-6 text-sm bg-gray-900/60 px-4 py-2 rounded border border-gray-800">
      <div>FPS Físico: <span id="stat-fps" class="font-bold text-green-400">0.0</span></div>
      <div>Passos Totais: <span id="stat-steps" class="font-bold text-yellow-400">0</span></div>
      <div>Latência P50: <span id="stat-lat-p50" class="font-bold text-cyan-400">0.0 µs</span></div>
      <div class="flex items-center gap-1.5">
        <span id="status-indicator" class="status-dot bg-green-500 animate-pulse"></span>
        <span id="stat-status" class="px-2 py-0.5 rounded text-xs bg-green-950 text-green-300 border border-green-800 font-semibold">STREAMING</span>
      </div>
    </div>
  </header>

  <!-- Grid Principal (3 Fases) -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
    <!-- Fase 1: Reflexo & Decisão -->
    <div class="card p-4">
      <div class="flex justify-between items-center mb-2 border-b border-gray-800 pb-1">
        <h2 class="text-sm font-semibold text-green-400">FASE 1: LATÊNCIA & GATING</h2>
        <span class="text-xs text-gray-500 font-mono">CPU Reflex</span>
      </div>
      <div class="h-48"><canvas id="chartLatency"></canvas></div>
      <div class="grid grid-cols-2 gap-2 mt-4 text-xs pt-2 border-t border-gray-800/60">
        <div>Confiança Média: <span id="stat-conf" class="font-bold text-green-300">--</span></div>
        <div>Incerteza Média: <span id="stat-unc" class="font-bold text-yellow-300">--</span></div>
      </div>
    </div>

    <!-- Fase 2: Performance no Ambiente -->
    <div class="card p-4">
      <div class="flex justify-between items-center mb-2 border-b border-gray-800 pb-1">
        <h2 class="text-sm font-semibold text-yellow-400">FASE 2: RECOMPENSA & FPS</h2>
        <span class="text-xs text-gray-500 font-mono">Environment Rollout</span>
      </div>
      <div class="h-48"><canvas id="chartReward"></canvas></div>
      <div class="grid grid-cols-2 gap-2 mt-4 text-xs pt-2 border-t border-gray-800/60">
        <div>Retorno Médio (20 ep): <span id="stat-ret20" class="font-bold text-yellow-300">--</span></div>
        <div>Total Episódios: <span id="stat-eps" class="font-bold text-gray-300">--</span></div>
      </div>
    </div>

    <!-- Fase 3: Estabilidade do PPO -->
    <div class="card p-4">
      <div class="flex justify-between items-center mb-2 border-b border-gray-800 pb-1">
        <h2 class="text-sm font-semibold text-purple-400">FASE 3: CONVERGÊNCIA PPO</h2>
        <span class="text-xs text-gray-500 font-mono">Recurrent BPTT</span>
      </div>
      <div class="h-48"><canvas id="chartLoss"></canvas></div>
      <div class="grid grid-cols-2 gap-2 mt-4 text-xs pt-2 border-t border-gray-800/60">
        <div>Policy Loss: <span id="stat-ploss" class="font-bold text-purple-300">--</span></div>
        <div>Value Loss: <span id="stat-vloss" class="font-bold text-pink-300">--</span></div>
      </div>
    </div>
  </div>

  <!-- Barra de Normas de Gradiente por Módulo -->
  <div class="card p-4">
    <div class="flex justify-between items-center mb-2 border-b border-gray-800 pb-1">
      <h2 class="text-sm font-semibold text-pink-400">ESTABILIDADE DE GRADIENTES (||∇|| POR MÓDULO)</h2>
      <span class="text-xs text-gray-500 font-mono">Backpropagation Health</span>
    </div>
    <div class="h-40"><canvas id="chartGrads"></canvas></div>
  </div>

  <script>
    // Configurações comuns dos gráficos Chart.js
    const chartOpts = {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: {
          labels: { color: '#94a3b8', font: { family: 'monospace', size: 10 } }
        }
      },
      scales: {
        x: { display: false },
        y: {
          ticks: { color: '#64748b', font: { family: 'monospace', size: 10 } },
          grid: { color: '#1e293b' }
        }
      }
    };

    // 1. Gráfico de Latência P50 / P99
    const ctxLat = document.getElementById('chartLatency').getContext('2d');
    const chartLat = new Chart(ctxLat, {
      type: 'line',
      data: {
        labels: Array(30).fill(''),
        datasets: [
          { label: 'P50 (µs)', data: [], borderColor: '#38bdf8', borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
          { label: 'P99 (µs)', data: [], borderColor: '#f43f5e', borderWidth: 1.5, pointRadius: 0, tension: 0.2 }
        ]
      },
      options: chartOpts
    });

    // 2. Gráfico de Retorno Acumulado
    const ctxRew = document.getElementById('chartReward').getContext('2d');
    const chartRew = new Chart(ctxRew, {
      type: 'line',
      data: {
        labels: Array(30).fill(''),
        datasets: [
          { label: 'Retorno Médio', data: [], borderColor: '#facc15', backgroundColor: 'rgba(250, 204, 21, 0.1)', fill: true, pointRadius: 0, tension: 0.2 }
        ]
      },
      options: chartOpts
    });

    // 3. Gráfico de Perdas PPO
    const ctxLoss = document.getElementById('chartLoss').getContext('2d');
    const chartLoss = new Chart(ctxLoss, {
      type: 'line',
      data: {
        labels: Array(30).fill(''),
        datasets: [
          { label: 'Policy Loss', data: [], borderColor: '#a855f7', borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
          { label: 'Value Loss', data: [], borderColor: '#ec4899', borderWidth: 1.5, pointRadius: 0, tension: 0.2 }
        ]
      },
      options: chartOpts
    });

    // 4. Gráfico de Normas de Gradiente por Bloco
    const ctxGrads = document.getElementById('chartGrads').getContext('2d');
    const chartGrads = new Chart(ctxGrads, {
      type: 'bar',
      data: {
        labels: ['FrontEnd (Perception)', 'Trunk (Recurrent Core)', 'PolicyHead (Action)'],
        datasets: [{
          label: 'Norma Euclidiana ||∇||',
          data: [0, 0, 0],
          backgroundColor: ['#38bdf8', '#818cf8', '#c084fc'],
          borderWidth: 1,
          borderColor: '#1e293b'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#94a3b8', font: { family: 'monospace', size: 11 } }, grid: { display: false } },
          y: { ticks: { color: '#64748b', font: { family: 'monospace' } }, grid: { color: '#1e293b' } }
        }
      }
    });

    // Conexão Server-Sent Events (SSE)
    let evtSource = null;
    function connectSSE() {
      evtSource = new EventSource('/stream');

      evtSource.onopen = () => {
        document.getElementById('stat-status').innerText = 'ONLINE';
        document.getElementById('stat-status').className = 'px-2 py-0.5 rounded text-xs bg-green-950 text-green-300 border border-green-800 font-semibold';
        document.getElementById('status-indicator').className = 'status-dot bg-green-500 animate-pulse';
      };

      evtSource.onerror = () => {
        document.getElementById('stat-status').innerText = 'RECONNECTING';
        document.getElementById('stat-status').className = 'px-2 py-0.5 rounded text-xs bg-yellow-950 text-yellow-300 border border-yellow-800 font-semibold';
        document.getElementById('status-indicator').className = 'status-dot bg-yellow-500 animate-pulse';
      };

      evtSource.onmessage = (e) => {
        const data = JSON.parse(e.data);

        document.getElementById('stat-fps').innerText = data.fps.toFixed(1);
        document.getElementById('stat-steps').innerText = (data.total_steps || 0).toLocaleString();
        document.getElementById('stat-lat-p50').innerText = data.latency_p50_us.toFixed(1) + ' µs';
        document.getElementById('stat-conf').innerText = (data.mean_confidence * 100).toFixed(1) + '%';
        document.getElementById('stat-unc').innerText = (data.mean_uncertainty * 100).toFixed(1) + '%';
        document.getElementById('stat-ret20').innerText = data.mean_return_20.toFixed(2);
        document.getElementById('stat-eps').innerText = (data.episodes_completed || 0).toLocaleString();
        document.getElementById('stat-ploss').innerText = (data.policy_loss >= 0 ? '+' : '') + data.policy_loss.toFixed(4);
        document.getElementById('stat-vloss').innerText = data.value_loss.toFixed(4);

        if (data.is_completed) {
          document.getElementById('stat-status').innerText = 'CONCLUÍDO';
          document.getElementById('stat-status').className = 'px-2 py-0.5 rounded text-xs bg-blue-950 text-blue-300 border border-blue-800 font-semibold';
          document.getElementById('status-indicator').className = 'status-dot bg-blue-500';
        } else {
          document.getElementById('stat-status').innerText = 'STREAMING';
          document.getElementById('stat-status').className = 'px-2 py-0.5 rounded text-xs bg-green-950 text-green-300 border border-green-800 font-semibold';
          document.getElementById('status-indicator').className = 'status-dot bg-green-500 animate-pulse';
        }

        // Atualiza arrays de gráfico (janela deslizante de 30 pontos)
        const pushPoint = (chart, dsIdx, val) => {
          const ds = chart.data.datasets[dsIdx].data;
          if (data.is_completed && ds.length > 0) return;
          ds.push(val);
          if (ds.length > 30) ds.shift();
        };

        pushPoint(chartLat, 0, data.latency_p50_us);
        pushPoint(chartLat, 1, data.latency_p99_us);
        chartLat.update('none');

        pushPoint(chartRew, 0, data.mean_return_20);
        chartRew.update('none');

        pushPoint(chartLoss, 0, data.policy_loss);
        pushPoint(chartLoss, 1, data.value_loss);
        chartLoss.update('none');

        if (data.grad_norms) {
          chartGrads.data.datasets[0].data = [
            data.grad_norms['FrontEnd'] || 0,
            data.grad_norms['Trunk'] || 0,
            data.grad_norms['PolicyHead'] || 0
          ];
          chartGrads.update('none');
        }
      };
    }

    connectSSE();
  </script>
</body>
</html>
"""


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Servidor TCP multi-thread para permitir múltiplos clientes SSE concorrentes sem bloquear."""

    allow_reuse_address = True
    daemon_threads = True


class TelemetryServer:
    """Servidor web de telemetria em tempo real com SSE (Server-Sent Events) sem dependências externas."""

    def __init__(
        self,
        tracker: LiveStatsTracker,
        host: str = "127.0.0.1",
        port: int = 8050,
        refresh_hz: float = 15.0,
        open_browser: bool = False,
    ) -> None:
        self.tracker = tracker
        self.host = host
        self.port = port
        self.refresh_interval = 1.0 / max(1.0, refresh_hz)
        self.open_browser = open_browser
        self.server: Optional[ThreadedTCPServer] = None
        self.thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Inicia o servidor HTTP em uma thread desacoplada de segundo plano."""
        if self._running:
            return

        tracker = self.tracker
        stop_event = self._stop_event
        refresh_interval = self.refresh_interval

        class TelemetryHandler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    try:
                        while not stop_event.is_set():
                            snap = tracker.snapshot()
                            # Garante que grad_norms estejam presentes no payload
                            snap["grad_norms"] = tracker.metrics.grad_norms
                            payload = f"data: {json.dumps(snap)}\n\n"
                            self.wfile.write(payload.encode("utf-8"))
                            self.wfile.flush()
                            time.sleep(refresh_interval)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass

                elif self.path == "/api/metrics":
                    # Endpoint REST auxiliar para testes e telemetria pura
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    snap = tracker.snapshot()
                    snap["grad_norms"] = tracker.metrics.grad_norms
                    self.wfile.write(json.dumps(snap).encode("utf-8"))

                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args) -> None:
                # Silencia mensagens de log de requests HTTP para manter o console limpo
                return

        self._stop_event.clear()
        self.server = ThreadedTCPServer((self.host, self.port), TelemetryHandler)
        self._running = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        dashboard_url = f"http://{self.host}:{self.port}"
        print(f"[✓] Painel Web em Tempo Real ativo em: {dashboard_url}")

        if self.open_browser:
            def _open() -> None:
                time.sleep(0.15)
                try:
                    import webbrowser
                    webbrowser.open(dashboard_url)
                except Exception:
                    pass

            threading.Thread(target=_open, daemon=True).start()

    def stop(self) -> None:
        """Encerra o servidor HTTP e libera a porta associada."""
        if self._running:
            self._stop_event.set()
            self._running = False
            if self.server is not None:
                self.server.shutdown()
                self.server.server_close()
                self.server = None

    def __enter__(self) -> "TelemetryServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
