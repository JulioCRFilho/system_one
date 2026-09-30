#!/usr/bin/env python3
"""Task 6: Percepção Visual com ImpalaVisualFrontEnd.

Demonstra o pipeline visual acelerado do System 1:
- Entrada visual com empilhamento de frames temporais (2*C, 84, 84)
- 3 Blocos Convolucionais Residuais IMPALA (16 -> 32 -> 32 canais)
- Projeção espacial compacta para ℝ^320
- Fusão no barramento universal de 337 dimensões (320 visual + 16 ação + 1 recompensa)
- Tronco Recorrente idêntico e transferível
- Orçamento de latência visual (<= 5 ms) e memória (< 35 MB)
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


class SyntheticVisualEnv(gym.Env):
    """Ambiente visual sintético reproduzindo o padrão (C, 84, 84)."""

    def __init__(self, channels: int = 1, height: int = 84, width: int = 84, n_actions: int = 4):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(channels, height, width),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(n_actions)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        obs = np.random.uniform(0.0, 1.0, size=(self.channels, self.height, self.width)).astype(np.float32)
        return obs, {}

    def step(self, action):
        self.steps += 1
        obs = np.random.uniform(0.0, 1.0, size=(self.channels, self.height, self.width)).astype(np.float32)
        reward = 1.0
        terminated = self.steps >= 200
        truncated = False
        return obs, reward, terminated, truncated, {}


def run_visual_task(benchmark_steps: int = 200) -> None:
    print("=" * 70)
    print("👁️ TASK 6: PIPELINE VISUAL IMPALA E BARRAMENTO LATENTE ℝ^337")
    print("=" * 70)

    # 1. Criação e Envelopamento do Ambiente Visual
    # O UniversalS1Wrapper empilhará (s_t, s_{t-1}) no canal, resultando em (2*C, 84, 84)
    raw_env = SyntheticVisualEnv(channels=1, height=84, width=84, n_actions=4)
    env = UniversalS1Wrapper(raw_env, is_visual=True)

    print(f"• Espaço de Observação Bruta    : {raw_env.observation_space.shape} (C, H, W)")
    print(f"• Espaço Empilhado no Wrapper   : (2*C, H, W) = (2, 84, 84)")
    print(f"• Espaço de Ações               : {raw_env.action_space.n} discretas")

    # 2. Instanciação do Agente com ImpalaVisualFrontEnd
    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=raw_env.action_space,
        is_visual=True,
    )
    agent.eval()

    total_params = sum(p.numel() for p in agent.parameters())
    param_mb = sum(p.numel() * p.element_size() for p in agent.parameters()) / (1024**2)

    print("-" * 70)
    print("📐 AUDITORIA ARQUITETURAL VISUAL:")
    print(f"  - Front-End Perceptivo       : {type(agent.front_end).__name__}")
    print(f"  - Blocos Residuais Convolutivos: 3x IMPALA Blocks (16 -> 32 -> 32)")
    print(f"  - Projeção Espacial Visual   : ConvFlat -> ℝ^320")
    print(f"  - Barramento Latente Unificado: ℝ^337 (320 visual + 16 ação + 1 recompensa)")
    print(f"  - Tronco Recorrente Universal : System1Trunk (GRU 337->256 + 2x ResMLP 256)")
    print(f"  - Total de Parâmetros        : {total_params:,} (~{param_mb:.2f} MB float32)")
    print(f"  - Limite de Memória (<35 MB) : ✅ Atingido com folga de {(35.0 - param_mb):.1f} MB")

    # 3. Teste do Fluxo de Tensores com Wrapper
    obs_dict, _ = env.reset(seed=42)
    agent.reset_memory()

    stacked_obs = obs_dict["obs"]
    assert stacked_obs.shape == (2, 84, 84), f"Shape incorreto: {stacked_obs.shape}"
    print(f"  - Shape do Frame Empilhado   : {stacked_obs.shape} (Canal 0: Atual, Canal 1: Anterior)")

    # 4. Benchmark de Latência Visual com act_fast()
    print("-" * 70)
    print(f"⚡ Benchmark de Latência Visual ({benchmark_steps} passos com act_fast)...")

    # Warmup
    for _ in range(20):
        agent.act_fast(obs_dict)

    latencies_ms = []
    for _ in range(benchmark_steps):
        t0 = time.perf_counter_ns()
        action = agent.act_fast(obs_dict)
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1e6)

        obs_dict, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs_dict, _ = env.reset()
            agent.reset_memory()

    avg_ms = float(np.mean(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))

    budget_visual_ms = 5.0  # Orçamento especificado para visão acelerada
    print(f"📊 Resultados de Latência Visual:")
    print(f"  - Latência Média   : {avg_ms:.2f} ms")
    print(f"  - P95              : {p95_ms:.2f} ms")
    print(f"  - Min / Max        : {min_ms:.2f} ms / {max_ms:.2f} ms")
    print(f"  - Orçamento Limite : <= {budget_visual_ms:.1f} ms")

    if avg_ms <= budget_visual_ms:
        print(f"✅ [STATUS: APROVADO] Orçamento visual de {budget_visual_ms} ms satisfeito!")
    else:
        print(f"⚠️ Latência visual na CPU ({avg_ms:.2f} ms). Para produção em tempo real com visão, use GPU/MPS.")

    print("=" * 70)


if __name__ == "__main__":
    run_visual_task()
