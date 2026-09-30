import time
from typing import Optional
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from system1_engine.telemetry.tracker import LiveStatsTracker


class S1LiveDashboard:
    """Renderizador de telemetria em tempo real com rich.live em thread desacoplada."""

    def __init__(
        self,
        tracker: LiveStatsTracker,
        refresh_rate_hz: int = 10,
        console: Optional[Console] = None,
    ) -> None:
        self.tracker = tracker
        self.refresh_interval = 1.0 / max(1, refresh_rate_hz)
        self.console = console or Console()
        self._live: Optional[Live] = None

    def generate_view(self) -> Layout:
        """Gera o layout com as três fases operacionais do System 1."""
        snap = self.tracker.snapshot()
        gn = self.tracker.metrics.grad_norms

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
        )
        layout["body"].split_row(
            Layout(name="phase1"),
            Layout(name="phase2"),
            Layout(name="phase3"),
        )

        # Header Superior
        header_text = Text(
            f"⚡ SYSTEM 1 ENGINE — PAINEL OPERACIONAL EM TEMPO REAL | Total Steps: {snap['total_steps']:,} | FPS: {snap['fps']:.1f}",
            style="bold cyan",
        )
        layout["header"].update(Panel(header_text, style="cyan"))

        # Painel Fase 1: Percepção e Inferência Reflexiva
        t1 = Table(expand=True, show_header=False)
        t1.add_column("Métrica", style="dim")
        t1.add_column("Valor", justify="right")
        t1.add_row("Latência P50", f"{snap['latency_p50_us']:.1f} µs")
        t1.add_row("Latência P99", f"{snap['latency_p99_us']:.1f} µs")
        t1.add_row("Confiança Média", f"{snap['mean_confidence'] * 100:.1f}%")
        t1.add_row("Incerteza Média", f"{snap['mean_uncertainty'] * 100:.1f}%")
        t1.add_row("Entropia Latente", f"{snap['mean_entropy']:.3f} nats")
        layout["phase1"].update(
            Panel(t1, title="[bold green]Fase 1: Inferência (Reflexo)[/]", border_style="green")
        )

        # Painel Fase 2: Ambiente & Rollout Físico
        t2 = Table(expand=True, show_header=False)
        t2.add_column("Métrica", style="dim")
        t2.add_column("Valor", justify="right")
        t2.add_row("Retorno Média (20 ep)", f"[bold yellow]{snap['mean_return_20']:.2f}[/]")
        t2.add_row("Episódios Concluídos", f"{snap['episodes_completed']}")
        t2.add_row("Throughput Físico", f"{snap['fps']:.1f} steps/s")
        layout["phase2"].update(
            Panel(t2, title="[bold yellow]Fase 2: Ambiente & Recompensa[/]", border_style="yellow")
        )

        # Painel Fase 3: Treinamento PPO & Saúde dos Gradientes
        t3 = Table(expand=True, show_header=False)
        t3.add_column("Métrica", style="dim")
        t3.add_column("Valor", justify="right")
        t3.add_row("Policy Loss", f"{snap['policy_loss']:+.4f}")
        t3.add_row("Value Loss", f"{snap['value_loss']:.4f}")
        t3.add_row("Clip Fraction", f"{snap['clip_frac'] * 100:.1f}%")
        t3.add_row("Learning Rate", f"{snap['lr']:.2e}")
        if gn:
            for block, norm in gn.items():
                t3.add_row(f"Grad {block}", f"{norm:.4f}")
        else:
            t3.add_row("Gradientes", "Aguardando batch...")
        layout["phase3"].update(
            Panel(t3, title="[bold magenta]Fase 3: Otimização PPO[/]", border_style="magenta")
        )

        return layout

    def start(self) -> None:
        """Inicia o streaming do dashboard ao vivo."""
        if self._live is None:
            self._live = Live(
                self.generate_view(),
                console=self.console,
                refresh_per_second=int(1.0 / self.refresh_interval),
                transient=False,
            )
            self._live.start()

    def stop(self) -> None:
        """Encerra o streaming do dashboard ao vivo."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def update(self) -> None:
        """Atualiza o conteúdo visual do dashboard."""
        if self._live is not None:
            self._live.update(self.generate_view())

    def render_once(self) -> None:
        """Imprime o layout uma única vez na console (ideal para ambientes headless ou relatórios)."""
        self.console.print(self.generate_view())

    def __enter__(self) -> "S1LiveDashboard":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
