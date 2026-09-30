from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class PhaseMetrics:
    """Estrutura com ring buffers (deques de tamanho fixo) em memória RAM para métricas O(1)."""

    # Fase 1: Percepção, Inferência & Decisão Reflexiva
    step_latencies_us: deque = field(default_factory=lambda: deque(maxlen=1000))
    uncertainties: deque = field(default_factory=lambda: deque(maxlen=1000))
    confidences: deque = field(default_factory=lambda: deque(maxlen=1000))
    entropies: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Fase 2: Ambiente Físico & Rollout GAE
    fps: float = 0.0
    current_episode_return: float = 0.0
    current_episode_length: int = 0
    episode_returns: deque = field(default_factory=lambda: deque(maxlen=100))
    episode_lengths: deque = field(default_factory=lambda: deque(maxlen=100))
    total_steps: int = 0

    # Fase 3: Otimização PPO & Saúde dos Gradientes
    policy_losses: deque = field(default_factory=lambda: deque(maxlen=100))
    value_losses: deque = field(default_factory=lambda: deque(maxlen=100))
    grad_norms: Dict[str, float] = field(default_factory=dict)
    clip_fractions: deque = field(default_factory=lambda: deque(maxlen=100))
    learning_rate: float = 0.0


class LiveStatsTracker:
    """Coletor centralizado de telemetria de alta frequência com overhead < 2 µs."""

    def __init__(self) -> None:
        self.metrics = PhaseMetrics()
        self._last_step_time = time.perf_counter_ns()
        self._fps_window: deque = deque(maxlen=100)

    def record_inference(
        self,
        latency_us: float,
        uncertainty: float,
        confidence: float,
        entropy: float,
    ) -> None:
        """Registra a fase de reflexo imediato com inserção O(1) sem alocação dinâmica."""
        m = self.metrics
        m.step_latencies_us.append(latency_us)
        m.uncertainties.append(uncertainty)
        m.confidences.append(confidence)
        m.entropies.append(entropy)

    def record_env_step(self, reward: float, done: bool) -> None:
        """Registra a fase de interação física com o ambiente."""
        now = time.perf_counter_ns()
        dt = (now - self._last_step_time) / 1e9
        self._last_step_time = now

        if dt > 0:
            self._fps_window.append(1.0 / dt)
            self.metrics.fps = float(np.mean(self._fps_window))

        m = self.metrics
        m.total_steps += 1
        m.current_episode_return += reward
        m.current_episode_length += 1

        if done:
            m.episode_returns.append(m.current_episode_return)
            m.episode_lengths.append(m.current_episode_length)
            m.current_episode_return = 0.0
            m.current_episode_length = 0

    def record_training_epoch(
        self,
        policy_loss: float,
        value_loss: float,
        clip_fraction: float,
        grad_norms: Dict[str, float],
        lr: float,
    ) -> None:
        """Registra a fase de atualização dos gradientes e saúde numérica."""
        m = self.metrics
        m.policy_losses.append(policy_loss)
        m.value_losses.append(value_loss)
        m.clip_fractions.append(clip_fraction)
        m.grad_norms = dict(grad_norms)
        m.learning_rate = lr

    def snapshot(self) -> Dict[str, Any]:
        """Extrai um resumo numérico instantâneo de todas as três fases."""
        m = self.metrics
        return {
            # Fase 1: Inferência & Ação
            "latency_p50_us": float(np.percentile(m.step_latencies_us, 50)) if m.step_latencies_us else 0.0,
            "latency_p99_us": float(np.percentile(m.step_latencies_us, 99)) if m.step_latencies_us else 0.0,
            "mean_uncertainty": float(np.mean(m.uncertainties)) if m.uncertainties else 0.0,
            "mean_confidence": float(np.mean(m.confidences)) if m.confidences else 0.0,
            "mean_entropy": float(np.mean(m.entropies)) if m.entropies else 0.0,
            # Fase 2: Ambiente & Rollout
            "fps": m.fps,
            "mean_return_20": float(np.mean(list(m.episode_returns)[-20:])) if m.episode_returns else 0.0,
            "total_steps": m.total_steps,
            "episodes_completed": len(m.episode_returns),
            "current_episode_return": m.current_episode_return,
            # Fase 3: Otimização PPO
            "policy_loss": m.policy_losses[-1] if m.policy_losses else 0.0,
            "value_loss": m.value_losses[-1] if m.value_losses else 0.0,
            "clip_frac": m.clip_fractions[-1] if m.clip_fractions else 0.0,
            "lr": m.learning_rate,
            "grad_norms": dict(m.grad_norms),
        }

    def reset(self) -> None:
        """Zera as métricas acumuladas."""
        self.metrics = PhaseMetrics()
        self._last_step_time = time.perf_counter_ns()
        self._fps_window.clear()
