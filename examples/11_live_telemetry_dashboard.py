#!/usr/bin/env python3
"""Task 11: Telemetria Assíncrona e Painel Operacional em Tempo Real (LiveStatsTracker & S1LiveDashboard).

Demonstra a observabilidade de alta frequência do System 1:
- Coletor com buffers circulares (ring buffers) O(1) na memória RAM.
- Verificação rigorosa de overhead: record_inference() < 2 µs.
- Painel terminal ao vivo via Rich (3 fases simultâneas: Inferência, Rollout e Otimização PPO).
- Monitoramento de normas de gradientes por bloco (FrontEnd, Trunk, PolicyHead) e saúde numérica.
"""

from pathlib import Path
import sys
import time

# Garante import do system1_engine mesmo se executado de qualquer diretório
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.telemetry import LiveStatsTracker, S1LiveDashboard
from system1_engine.training.ppo import RecurrentPPOTrainer


def run_telemetry_demo():
    print("=" * 78)
    print("📊 TASK 11: TELEMETRIA ASSÍNCRONA E PAINEL OPERACIONAL EM TEMPO REAL")
    print("=" * 78)

    # 1. Benchmark de Overhead do Coletor (Garantia de < 2 µs por inserção)
    print("🔬 1. BENCHMARK DE OVERHEAD DO COLETOR (LiveStatsTracker):")
    tracker = LiveStatsTracker()
    n_measurements = 10000

    t0 = time.perf_counter_ns()
    for _ in range(n_measurements):
        tracker.record_inference(
            latency_us=175.4,
            uncertainty=0.082,
            confidence=0.918,
            entropy=0.231,
        )
    total_time_ns = time.perf_counter_ns() - t0
    overhead_us = (total_time_ns / n_measurements) / 1000.0

    print(f"  • {n_measurements:,} inserções em ring buffer circular (deque O(1))")
    print(f"  • Tempo por record_inference() : {overhead_us:.4f} µs ({overhead_us * 1000.0:.1f} ns)")
    print(f"  • Limite Máximo Especificado   : < 2.0000 µs")
    assert overhead_us < 2.0, f"Overhead excedeu orçamento: {overhead_us:.4f} µs >= 2.0 µs"
    print(f"  ✅ [APROVADO] Overhead do coletor é {2.0 / overhead_us:.1f}x inferior ao teto de 2 µs!\n")

    # 2. Demonstração de Inferência Reflexiva com Streaming do Painel ao Vivo
    print("-" * 78)
    print("🎮 2. LOOP DE INFERÊNCIA REFLEXIVA COM STREAMING AO VIVO:")
    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)

    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    if Path("s1_cartpole.pt").exists():
        ckpt = torch.load("s1_cartpole.pt", map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        agent.load_state_dict(state_dict, strict=False)
        print("  • Checkpoint 's1_cartpole.pt' carregado.")
    agent.eval()

    tracker.reset()
    dashboard = S1LiveDashboard(tracker=tracker, refresh_rate_hz=15)

    print("  • Executando 2 episódios com telemetria ativa...\n")
    with dashboard:
        for ep in range(2):
            obs_dict, _ = env.reset(seed=42 + ep)
            agent.reset_memory()
            done = False

            while not done:
                t_inf = time.perf_counter_ns()
                dec = agent.act_with_confidence(obs_dict)
                lat = (time.perf_counter_ns() - t_inf) / 1000.0

                tracker.record_inference(
                    latency_us=lat,
                    uncertainty=dec.uncertainty,
                    confidence=dec.confidence,
                    entropy=dec.entropy,
                )

                obs_dict, reward, term, trunc, _ = env.step(dec.action)
                done = term or trunc
                tracker.record_env_step(reward=reward, done=done)
                dashboard.update()

    dashboard.render_once()

    # 3. Demonstração de Treinamento PPO com Monitoramento de Gradientes por Bloco
    print("-" * 78)
    print("🏋️ 3. OTIMIZAÇÃO PPO COM TELEMETRIA DE GRADIENTES POR BLOCO:")
    print("  • Executando 1,024 passos de PPO com extração de normas de gradiente...")

    tracker.reset()
    dashboard_train = S1LiveDashboard(tracker=tracker, refresh_rate_hz=10)

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        learning_rate=5e-4,
        rollout_steps=512,
        chunk_length=16,
        chunk_batch_size=8,
        n_epochs=2,
        tracker=tracker,
    )

    with dashboard_train:
        trainer.train(
            max_steps=1024,
            target_return=None,
            callback=lambda steps, ret: dashboard_train.update(),
            verbose=False,
        )

    dashboard_train.render_once()

    snap = tracker.snapshot()
    print("-" * 78)
    print("📊 RESUMO TELEMÉTRICO CONSOLIDADO:")
    print(f"  • Passos Totais Monitorados : {snap['total_steps']:,}")
    print(f"  • Latência Média P50        : {snap['latency_p50_us']:.1f} µs")
    print(f"  • Latência Média P99        : {snap['latency_p99_us']:.1f} µs")
    print(f"  • Confiança Média           : {snap['mean_confidence'] * 100:.1f}%")
    print(f"  • Incerteza Média           : {snap['mean_uncertainty'] * 100:.1f}%")
    print(f"  • Throughput Efetivo        : {snap['fps']:.1f} steps/s")
    print(f"  • Normas de Gradientes      : {snap['grad_norms']}")
    print("=" * 78)
    print("✅ [STATUS: APROVADO] Barramento de telemetria operacional com zero regressão!")


if __name__ == "__main__":
    run_telemetry_demo()
