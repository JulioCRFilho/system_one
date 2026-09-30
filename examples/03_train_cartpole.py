#!/usr/bin/env python3
"""Task 3: Treinamento do Zero via Recurrent PPO (CartPole-v1).

Demonstra o pipeline completo de treinamento do System 1 usando a implementação
pura em PyTorch do Recurrent PPO com truncamento por chunks e GAE.
"""

from pathlib import Path
import sys
import time

# Garante import do system1_engine mesmo se executado de qualquer diretório
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.transfer.manager import KnowledgeTransferManager


def train_agent(
    env_id: str = "CartPole-v1",
    max_steps: int = 35000,
    target_return: float = 475.0,
    save_path: str = "s1_cartpole_custom.pt",
    seed: int = 42,
) -> None:
    print("=" * 70)
    print(f"🏋️ TASK 3: TREINAMENTO DO ZERO COM RECURRENT PPO ({env_id})")
    print("=" * 70)

    torch.manual_seed(seed)

    # 1. Envelopamento do Ambiente com UniversalS1Wrapper
    # Garante cálculo de derivada diferencial Δs_t e histórico causal (a_{t-1}, r_{t-1})
    raw_env = gym.make(env_id)
    env = UniversalS1Wrapper(raw_env)

    # 2. Inicialização do Agente System 1
    # Front-end vetorial (337 dims) -> Tronco Recorrente GRU+ResMLP (256) -> Heads
    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
    )

    print("• Configuração do Agente:")
    print(f"  - Entrada: VectorFrontEnd -> Tronco Recorrente (337 -> 256) -> Heads")
    print(f"  - Parâmetros: {sum(p.numel() for p in agent.parameters()):,} tensores")

    # 3. Configuração do Treinador PPO Recorrente
    # Chunk length = 16 (BPTT temporal sobre sequências sem vazamento)
    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        value_coef=0.05,
        entropy_coef=0.001,
        rollout_steps=1024,
        chunk_length=16,
        chunk_batch_size=16,
        n_epochs=4,
    )

    print("• Hiperparâmetros de Treino:")
    print(f"  - Rollout Steps : {trainer.rollout_steps}")
    print(f"  - BPTT Chunks   : Length={trainer.chunk_length}, BatchSize={trainer.chunk_batch_size}")
    print(f"  - Taxa de Aprendizado (LR): 1e-3 | Épocas por Rollout: 4")
    print(f"  - Meta de Retorno: {target_return} (Critério de parada antecipada)")
    print("-" * 70)

    t0 = time.time()
    final_return = trainer.train(
        max_steps=max_steps,
        target_return=target_return,
        verbose=True,
    )
    t_total = time.time() - t0

    print("-" * 70)
    print(f"⏱️ Treinamento concluído em {t_total:.1f}s ({trainer.total_steps} passos coletados)")
    print(f"🎯 Média Móvel de Recompensa Final: {final_return:.2f}")

    # 4. Salvando Checkpoint Consolidado
    KnowledgeTransferManager.save_checkpoint(
        agent=agent,
        checkpoint_path=save_path,
        extra_info={
            "env_id": env_id,
            "final_return": final_return,
            "steps": trainer.total_steps,
            "elapsed_seconds": t_total,
        },
    )
    print(f"💾 Checkpoint persistido com sucesso em: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    train_agent()
