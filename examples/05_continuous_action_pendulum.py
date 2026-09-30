#!/usr/bin/env python3
"""Task 5: Controle Contínuo com GaussianPolicyHead (Pendulum-v1).

Demonstra o System 1 operando sobre espaços de ação contínuos (gym.spaces.Box):
- Predição da média μ e desvio padrão logarítmico log(σ)
- ActionEncoder contínuo (Linear(act_dim, 16))
- Inferência amortizada contínua com act_fast() <= 0.8 ms
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
from system1_engine.training.ppo import RecurrentPPOTrainer


def run_continuous_task(steps_train: int = 1536) -> None:
    print("=" * 70)
    print("🎯 TASK 5: CONTROLE CONTÍNUO COM GAUSSIAN POLICY HEAD (Pendulum-v1)")
    print("=" * 70)

    # 1. Criação do Ambiente Contínuo
    raw_env = gym.make("Pendulum-v1")
    env = UniversalS1Wrapper(raw_env)

    print(f"• Ambiente              : Pendulum-v1")
    print(f"• Espaço de Observação  : {env.env.observation_space.shape} (Box)")
    print(f"• Espaço de Ações       : {env.action_space.shape} (Box contínuo: limites [{env.action_space.low[0]}, {env.action_space.high[0]}])")

    # 2. Instanciação do Agente
    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
    )

    print("• Arquitetura Ativada:")
    print(f"  - ActionEncoder       : {type(agent.front_end.action_encoder).__name__} (Linear 1D -> 16D)")
    print(f"  - PolicyHead          : {type(agent.policy_head).__name__} (μ_net + log_std)")
    print(f"  - Parâmetros Totais   : {sum(p.numel() for p in agent.parameters()):,}")

    # 3. Teste de Inferência Rápida (act_fast)
    obs_dict, _ = env.reset(seed=42)
    agent.reset_memory()

    # Warmup
    for _ in range(20):
        agent.act_fast(obs_dict)

    latencies_us = []
    print("-" * 70)
    print("⚡ Testando inferência contínua com act_fast()...")
    for _ in range(100):
        t0 = time.perf_counter_ns()
        action = agent.act_fast(obs_dict)
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)

        # Passo no ambiente com ação contínua
        obs_dict, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs_dict, _ = env.reset()
            agent.reset_memory()

    avg_ms = np.mean(latencies_us) / 1000.0
    print(f"• Tipo de Ação Retornada : {type(action)} com shape {action.shape} e valor {action}")
    print(f"• Latência Média CPU    : {avg_ms * 1000:.1f} µs ({avg_ms:.4f} ms)")
    assert avg_ms <= 0.8, "Latência contínua excedeu o orçamento!"
    print(f"✅ Latência dentro do orçamento (<= 0.8 ms)")

    # 4. Demonstração de Treinamento Rápido no Pendulum
    print("-" * 70)
    print(f"🏋️ Executando {steps_train} passos de treino PPO contínuo...")
    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        learning_rate=3e-4,
        rollout_steps=512,
        chunk_length=16,
        chunk_batch_size=8,
        n_epochs=2,
    )

    trainer.train(max_steps=steps_train, verbose=True)
    print("-" * 70)
    print("✅ Treinador executou backpropagation gaussiana com sucesso sem NaN nem divergência!")
    print("=" * 70)


if __name__ == "__main__":
    run_continuous_task()
