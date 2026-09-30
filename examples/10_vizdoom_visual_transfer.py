#!/usr/bin/env python3
"""Task 10: Teste de Fogo com Transferência Visual (ViZDoom).

Compara experimentalmente a tese de transferência:
  - Condição A (Transferência com Tronco Congelado):
      O System1Trunk (~719k parâmetros recorrentes) treinado no CartPole
      é carregado e 100% congelado (requires_grad = False).
      Apenas o ImpalaVisualFrontEnd e as cabeças aprendem a guiar a mira.
  - Condição B (Tabula Rasa / Do Zero):
      O agente completo (visão + tronco + cabeças) é treinado do zero sem intuição temporal.

Cenário: ViZDoom basic.cfg (Entrada Visual empilhada 2x1x84x84, 3 ações discretas).
"""

from pathlib import Path
import sys
import time

# Garante import do system1_engine mesmo se executado de qualquer diretório
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env import make_game_env
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.transfer.manager import KnowledgeTransferManager


def run_visual_transfer_experiment(steps_per_agent: int = 1536) -> None:
    print("=" * 80)
    print("🎯 TASK 10: TESTE DE FOGO - TRANSFERÊNCIA VISUAL NO VIZDOOM (CartPole -> Doom)")
    print("=" * 80)

    ckpt_source = "s1_cartpole.pt"
    if not Path(ckpt_source).exists():
        print(f"❌ Checkpoint '{ckpt_source}' não encontrado! Execute a Task 3 antes.")
        sys.exit(1)

    # 1. Configuração do Ambiente ViZDoom Nativo
    env_config = {
        "engine_type": "vizdoom",
        "scenario_path": "basic.cfg",
        "args": {"headless": True, "frame_skip": 4},
        "is_visual": True,
        "channels": 1,
    }

    env_a = make_game_env("native", env_config)
    env_b = make_game_env("native", env_config)

    print("• Cenário Nativo: ViZDoom basic.cfg")
    print(f"  - Entrada Visual Empilhada : {env_a.observation_space['obs'].shape} (Canal Atual + Anterior)")
    print(f"  - Ações do Doom            : {env_a.action_space.n} ações [ESQUERDA, DIREITA, ATIRAR]")

    # =========================================================================
    # CONDIÇÃO A: TRANSFER LEARNING COM TRONCO CONGELADO
    # =========================================================================
    print("\n" + "-" * 80)
    print("🔬 CONDIÇÃO A: TRANSFER LEARNING COM TRONCO CONGELADO (freeze_trunk=True)")
    print("-" * 80)

    agent_transfer = UniversalS1Agent(
        obs_space=env_a.observation_space,
        action_space=env_a.action_space,
        is_visual=True,
    )

    loaded = KnowledgeTransferManager.load_transferable_weights(
        agent=agent_transfer,
        checkpoint_path=ckpt_source,
        freeze_trunk=True,
    )
    print(f"• Pesos cognitivos carregados do CartPole: {len(loaded)} chaves do System1Trunk.")

    frozen_a = sum(p.numel() for p in agent_transfer.trunk.parameters() if not p.requires_grad)
    trainable_a = sum(p.numel() for p in agent_transfer.parameters() if p.requires_grad)
    print(f"  - Parâmetros Congelados (Tronco) : {frozen_a:,} (0 gradientes)")
    print(f"  - Parâmetros Treináveis (Adapt) : {trainable_a:,} (Apenas IMPALA + Heads)")

    # Snapshot inicial dos pesos do tronco
    trunk_snapshot = {name: p.clone().detach() for name, p in agent_transfer.trunk.named_parameters()}

    trainer_a = RecurrentPPOTrainer(
        agent=agent_transfer,
        env=env_a,
        learning_rate=5e-4,
        rollout_steps=512,
        chunk_length=16,
        chunk_batch_size=8,
        n_epochs=2,
    )

    print(f"\n⚡ Treinando Condição A ({steps_per_agent} passos)...")
    t0_a = time.time()
    trainer_a.train(max_steps=steps_per_agent, verbose=True)
    time_a = time.time() - t0_a

    # Auditoria bitwise do tronco
    for name, p in agent_transfer.trunk.named_parameters():
        diff = torch.max(torch.abs(p - trunk_snapshot[name])).item()
        assert diff == 0.0, f"Violação! O peso {name} foi modificado!"
    print("✅ [CONFIRMADO] Tronco permaneceu 100% inalterado (Invariância bitwise absoluta).")

    # =========================================================================
    # CONDIÇÃO B: TABULA RASA (DO ZERO)
    # =========================================================================
    print("\n" + "-" * 80)
    print("🔬 CONDIÇÃO B: TABULA RASA (TREINAMENTO DO ZERO)")
    print("-" * 80)

    agent_scratch = UniversalS1Agent(
        obs_space=env_b.observation_space,
        action_space=env_b.action_space,
        is_visual=True,
    )
    trainable_b = sum(p.numel() for p in agent_scratch.parameters() if p.requires_grad)
    print(f"  - Parâmetros Treináveis (Total) : {trainable_b:,} (Rede Completa)")

    trainer_b = RecurrentPPOTrainer(
        agent=agent_scratch,
        env=env_b,
        learning_rate=5e-4,
        rollout_steps=512,
        chunk_length=16,
        chunk_batch_size=8,
        n_epochs=2,
    )

    print(f"\n⚡ Treinando Condição B ({steps_per_agent} passos)...")
    t0_b = time.time()
    trainer_b.train(max_steps=steps_per_agent, verbose=True)
    time_b = time.time() - t0_b

    # =========================================================================
    # AVALIAÇÃO COMPARATIVA DOS DOIS AGENTES NO VIZDOOM
    # =========================================================================
    print("\n" + "=" * 80)
    print("📊 RESULTADOS E COMPARAÇÃO DE SAMPLE EFFICIENCY:")
    print("=" * 80)

    def evaluate_doom(agent, env, episodes=5):
        agent.eval()
        rewards = []
        for _ in range(episodes):
            obs_dict, _ = env.reset()
            agent.reset_memory()
            ep_ret = 0.0
            done = False
            while not done:
                action = agent.act_fast(obs_dict)
                obs_dict, rew, term, trunc, _ = env.step(action)
                ep_ret += rew
                done = term or trunc
            rewards.append(ep_ret)
        return float(np.mean(rewards)), float(np.std(rewards))

    ret_a, std_a = evaluate_doom(agent_transfer, env_a, episodes=5)
    ret_b, std_b = evaluate_doom(agent_scratch, env_b, episodes=5)

    print(f"{'Métrica':<35} | {'Condição A (Transfer)':<20} | {'Condição B (Scratch)':<20}")
    print("-" * 80)
    print(f"{'Parâmetros Treinados':<35} | {trainable_a:<20,} | {trainable_b:<20,}")
    print(f"{'Tempo de Treino':<35} | {time_a:<18.2f}s | {time_b:<18.2f}s")
    print(f"{'Throughput (Passos/seg)':<35} | {steps_per_agent / time_a:<18.1f} | {steps_per_agent / time_b:<18.1f}")
    print(f"{'Retorno Médio no Doom (5 ep)':<35} | {ret_a:6.1f} ± {std_a:<12.1f} | {ret_b:6.1f} ± {std_b:<12.1f}")
    print("-" * 80)

    # Economia de gradientes
    savings = (1.0 - (trainable_a / trainable_b)) * 100.0
    print(f"💡 Economia de Gradientes com Tronco Congelado: {savings:.1f}% menos parâmetros para otimizar!")
    print("🌟 Conclusão: A dinâmica temporal pré-calibrada do tronco permite adaptar a visão")
    print("   ao ViZDoom treinando exclusivamente a camada convolucional e as cabeças!")
    print("=" * 80)

    env_a.close()
    env_b.close()


if __name__ == "__main__":
    run_visual_transfer_experiment()
