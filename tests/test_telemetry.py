from collections import deque
import time
import gymnasium as gym
import numpy as np
import pytest
import torch
from rich.layout import Layout

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.telemetry.dashboard import S1LiveDashboard
from system1_engine.telemetry.tracker import LiveStatsTracker, PhaseMetrics
from system1_engine.training.ppo import RecurrentPPOTrainer


def test_livestats_tracker_initialization_and_ring_buffers():
    """Garante que os buffers circulares possuem tamanhos fixos e tempo O(1)."""
    tracker = LiveStatsTracker()
    m = tracker.metrics

    assert isinstance(m.step_latencies_us, deque)
    assert m.step_latencies_us.maxlen == 1000
    assert m.uncertainties.maxlen == 1000
    assert m.confidences.maxlen == 1000
    assert m.entropies.maxlen == 1000
    assert m.episode_returns.maxlen == 100
    assert m.policy_losses.maxlen == 100

    # Inserir mais elementos que maxlen e garantir que o teto é respeitado
    for i in range(1500):
        tracker.record_inference(latency_us=float(i), uncertainty=0.1, confidence=0.9, entropy=0.5)

    assert len(m.step_latencies_us) == 1000
    assert m.step_latencies_us[-1] == 1499.0
    assert m.step_latencies_us[0] == 500.0


def test_livestats_tracker_record_inference_overhead():
    """Garante rigorosamente que o overhead por chamada de record_inference é < 2.0 µs."""
    tracker = LiveStatsTracker()
    n_iters = 5000

    # Warmup
    for _ in range(500):
        tracker.record_inference(180.0, 0.15, 0.85, 0.4)

    t0 = time.perf_counter_ns()
    for _ in range(n_iters):
        tracker.record_inference(175.2, 0.12, 0.88, 0.35)
    total_time_ns = time.perf_counter_ns() - t0

    overhead_us = (total_time_ns / n_iters) / 1000.0
    assert overhead_us < 2.0, f"Overhead de telemetria ({overhead_us:.4f} µs) excedeu o teto de 2.0 µs!"


def test_livestats_tracker_env_step_and_fps():
    """Valida a transição de episódios, acúmulo de retorno e cálculo de FPS."""
    tracker = LiveStatsTracker()

    tracker.record_env_step(reward=1.0, done=False)
    tracker.record_env_step(reward=2.0, done=False)
    tracker.record_env_step(reward=3.0, done=True)

    assert tracker.metrics.total_steps == 3
    assert len(tracker.metrics.episode_returns) == 1
    assert tracker.metrics.episode_returns[0] == 6.0
    assert tracker.metrics.current_episode_return == 0.0

    # Inicia próximo episódio
    tracker.record_env_step(reward=5.0, done=False)
    assert tracker.metrics.current_episode_return == 5.0
    assert tracker.metrics.total_steps == 4


def test_livestats_tracker_training_epoch_and_snapshot():
    """Valida a consolidação dos gradientes, perdas e snapshot com percentis."""
    tracker = LiveStatsTracker()

    # Registra inferências
    for lat in [100.0, 200.0, 300.0, 400.0, 500.0]:
        tracker.record_inference(latency_us=lat, uncertainty=0.2, confidence=0.8, entropy=0.45)

    tracker.record_env_step(reward=10.0, done=True)

    grad_norms = {"FrontEnd": 0.123, "Trunk": 1.456, "PolicyHead": 2.789}
    tracker.record_training_epoch(
        policy_loss=0.045,
        value_loss=1.234,
        clip_fraction=0.15,
        grad_norms=grad_norms,
        lr=3e-4,
    )

    snap = tracker.snapshot()

    assert snap["latency_p50_us"] == 300.0
    assert snap["latency_p99_us"] >= 450.0
    assert snap["mean_uncertainty"] == pytest.approx(0.2, abs=1e-5)
    assert snap["mean_confidence"] == pytest.approx(0.8, abs=1e-5)
    assert snap["mean_entropy"] == pytest.approx(0.45, abs=1e-5)
    assert snap["mean_return_20"] == 10.0
    assert snap["policy_loss"] == 0.045
    assert snap["value_loss"] == 1.234
    assert snap["clip_frac"] == 0.15
    assert snap["lr"] == 3e-4
    assert snap["grad_norms"] == grad_norms


def test_s1_live_dashboard_generate_view_and_render():
    """Valida a geração da árvore de layout do Rich com as 3 fases."""
    tracker = LiveStatsTracker()
    tracker.record_inference(latency_us=185.0, uncertainty=0.1, confidence=0.9, entropy=0.2)
    tracker.record_env_step(reward=20.0, done=True)
    tracker.record_training_epoch(
        policy_loss=0.01,
        value_loss=0.5,
        clip_fraction=0.05,
        grad_norms={"FrontEnd": 0.1, "Trunk": 0.5, "PolicyHead": 1.0},
        lr=1e-3,
    )

    dashboard = S1LiveDashboard(tracker=tracker)
    layout = dashboard.generate_view()

    assert isinstance(layout, Layout)
    assert "header" in [child.name for child in layout.children]
    assert "body" in [child.name for child in layout.children]

    body_children = [c.name for c in layout["body"].children]
    assert "phase1" in body_children
    assert "phase2" in body_children
    assert "phase3" in body_children

    # Renderização estática segura
    dashboard.render_once()


def test_trainer_integration_with_telemetry():
    """Garante que o RecurrentPPOTrainer popula o LiveStatsTracker automaticamente."""
    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)
    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)

    tracker = LiveStatsTracker()
    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        rollout_steps=128,
        chunk_length=16,
        chunk_batch_size=4,
        n_epochs=1,
        tracker=tracker,
    )

    trainer.train(max_steps=128, verbose=False)

    assert tracker.metrics.total_steps == 128
    assert len(tracker.metrics.step_latencies_us) == 128
    assert len(tracker.metrics.confidences) == 128
    assert len(tracker.metrics.uncertainties) == 128
    assert len(tracker.metrics.policy_losses) > 0
    assert "FrontEnd" in tracker.metrics.grad_norms
    assert "Trunk" in tracker.metrics.grad_norms
    assert "PolicyHead" in tracker.metrics.grad_norms


def test_livestats_tracker_curriculum_depth_peak():
    """Valida o isolamento e reset do pico de retorno para a profundidade atual do currículo."""
    tracker = LiveStatsTracker()

    # Episódio em D1 com retorno alto
    tracker.record_env_step(reward=10.0, done=False, info={"current_depth": 1, "max_depth": 10})
    tracker.record_env_step(reward=5.0, done=True, info={"current_depth": 1, "max_depth": 10})

    snap = tracker.snapshot()
    assert snap["best_return"] == 15.0
    assert snap["curriculum_depth"] == 1
    assert snap["curriculum_depth_best_return"] == 15.0

    # Promovido para D2: pico global continua 15.0, mas pico do nível D2 é resetado e atualizado
    tracker.record_env_step(reward=2.0, done=False, info={"current_depth": 2, "max_depth": 10})
    tracker.record_env_step(reward=3.0, done=True, info={"current_depth": 2, "max_depth": 10})

    snap2 = tracker.snapshot()
    assert snap2["best_return"] == 15.0  # Pico global permanece
    assert snap2["curriculum_depth"] == 2
    assert snap2["curriculum_depth_best_return"] == 5.0  # Pico isolado do D2!
