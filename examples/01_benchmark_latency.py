#!/usr/bin/env python3
"""Task 1: Benchmark de Latência de Inferência CPU (act_fast).

Mede a latência em microssegundos/milissegundos da passagem direta determinística
do System 1 em CPU pura, verificando o orçamento rígido de <= 0.8 ms.
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


def run_benchmark(steps: int = 1000) -> None:
    print("=" * 70)
    print("🚀 TASK 1: BENCHMARK DE LATÊNCIA DE INFERÊNCIA CPU (act_fast)")
    print("=" * 70)

    # 1. Instanciação do Ambiente e Agente
    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)
    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
    )
    agent.eval()

    # Contagem de Parâmetros e Memória
    total_params = sum(p.numel() for p in agent.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in agent.parameters())
    print(f"• Parâmetros Totais: {total_params:,} ({param_bytes / (1024**2):.2f} MB float32)")
    print(f"• Hardware de Execução: CPU ({torch.get_num_threads()} threads PyTorch)")

    # 2. Reset inicial e Warmup da CPU/Cache
    obs_dict, _ = env.reset(seed=42)
    agent.reset_memory()

    print("• Aquecendo caches de CPU (50 forward passes)...")
    for _ in range(50):
        agent.act_fast(obs_dict)

    # 3. Benchmark de Precisão
    latencies_us = []
    for _ in range(steps):
        t0 = time.perf_counter_ns()
        action = agent.act_fast(obs_dict)
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)  # microssegundos

    latencies_ms = np.array(latencies_us) / 1000.0

    avg_ms = float(np.mean(latencies_ms))
    med_ms = float(np.median(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))
    std_ms = float(np.std(latencies_ms))

    budget_ms = 0.8
    speedup = budget_ms / avg_ms

    print("-" * 70)
    print(f"📊 RESULTADOS DO BENCHMARK ({steps:,} passos sequenciais):")
    print(f"  - Latência Média   : {avg_ms * 1000:.1f} µs ({avg_ms:.4f} ms)")
    print(f"  - Latência Mediana : {med_ms * 1000:.1f} µs ({med_ms:.4f} ms)")
    print(f"  - P95 (95th %)     : {p95_ms * 1000:.1f} µs ({p95_ms:.4f} ms)")
    print(f"  - P99 (99th %)     : {p99_ms * 1000:.1f} µs ({p99_ms:.4f} ms)")
    print(f"  - Min / Max        : {min_ms:.4f} ms / {max_ms:.4f} ms")
    print(f"  - Desvio Padrão    : {std_ms:.4f} ms")
    print(f"  - Orçamento Limite : <= {budget_ms:.4f} ms")
    print(f"  - Fator de Folga   : {speedup:.1f}x mais rápido que o orçamento máximo")
    print("-" * 70)

    if avg_ms <= budget_ms:
        print("✅ [STATUS: APROVADO] Orçamento de latência satisfeito com sucesso!")
    else:
        print("❌ [STATUS: REPROVADO] Orçamento de latência excedido.")
        sys.exit(1)


if __name__ == "__main__":
    run_benchmark()
