#!/usr/bin/env python3
"""Task 13: Treinamento do System 1 no Cubo Mágico (Ações Atômicas & Macro-Ações Pré-Definidas).

Demonstra:
1. O catálogo de macro-ações pré-definidas (Hierarchical RL / Options Framework).
2. Treinamento de reflexo amortizado com Recurrent PPO sobre observação vetorial one-hot (324 dims).
3. Avaliação passo a passo com telemetria de incerteza (Confidence Gating).
4. Renderização visual 2D Net integrada ao HUD.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

# Garante import do system1_engine de qualquer diretório
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.adapters.rubiks import RubiksCubeCore, RubiksCubeMacroEnv, RubiksCubeEnv
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.telemetry.tracker import LiveStatsTracker
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.transfer.manager import KnowledgeTransferManager


def display_macro_catalog() -> None:
    print("\n" + "=" * 75)
    print("🧩 CATÁLOGO DE MACRO-AÇÕES E ALGORITMOS PRÉ-DEFINIDOS (SYSTEM 1)")
    print("=" * 75)
    print(f"{'Ação':<4} | {'Nome da Macro':<18} | {'Sequência de Giros':<28} | {'Efeito Controlado'}")
    print("-" * 75)
    descriptions = {
        "SEXY_MOVE_R": "Trigger Direito (troca/orienta quinas)",
        "SEXY_MOVE_L": "Trigger Esquerdo (simetria)",
        "SUNE": "Orientação de cantos amarelos (topo)",
        "ANTI_SUNE": "Orientação inversa de cantos (topo)",
        "T_PERM": "Permutação de quinas/meios da última camada",
        "INSERT_EDGE_R": "Insere meio na segunda camada (direita)",
        "INSERT_EDGE_L": "Insere meio na segunda camada (esquerda)",
        "YELLOW_CROSS": "Fru-Ruf (cria a cruz amarela no topo)",
        "ROTATE_Y": "Giro do cubo todo no eixo Y (+90°)",
        "ROTATE_Y_PRIME": "Giro do cubo todo no eixo Y (-90°)",
        "U_TURN": "Giro da camada superior (alinhamento)",
        "U_PRIME_TURN": "Giro inverso da camada superior",
    }
    for idx, (name, seq) in enumerate(RubiksCubeCore.MACRO_ACTIONS.items()):
        seq_str = " ".join(seq)
        if len(seq_str) > 26:
            seq_str = seq_str[:23] + "..."
        desc = descriptions.get(name, "Movimento pré-definido")
        print(f"{idx:<4} | {name:<18} | {seq_str:<28} | {desc}")
    print("=" * 75 + "\n")


def train_rubiks(
    mode: str = "macro",
    steps: int = 15000,
    scramble_depth: int = 2,
    lr: float = 1e-3,
    save_path: Optional[str] = None,
    eval_episodes: int = 3,
) -> None:
    env_id = "RubiksCubeMacro-v0" if mode == "macro" else "RubiksCube-v0"
    default_save = f"s1_{'rubiks_macro' if mode == 'macro' else 'rubiks_atomic'}_trained.pt"
    save_file = save_path or default_save

    print("=" * 75)
    print(f"🎲 TREINAMENTO DO SYSTEM 1 NO CUBO MÁGICO ({mode.upper()})")
    print(f"Ambiente: {env_id} | Scramble Depth: {scramble_depth} | Passos Máximos: {steps}")
    print("=" * 75)

    if mode == "macro":
        display_macro_catalog()
        raw_env = RubiksCubeMacroEnv(scramble_depth=scramble_depth)
    else:
        raw_env = RubiksCubeEnv(scramble_depth=scramble_depth)

    # Universal Wrapper (adiciona derivadas Δs_t e histórico causal a_{t-1}, r_{t-1})
    env = UniversalS1Wrapper(raw_env)

    # UniversalS1Agent
    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
    )

    print("• Configuração do Agente Universal S1:")
    print(f"  - Espaço de Observação: VectorFrontEnd ({env.observation_space['obs'].shape[0]} dims one-hot)")
    print(f"  - Tronco Recorrente: GRU + 2x ResMLP (337 -> 256 latente)")
    print(f"  - Espaço de Ação: CategoricalPolicyHead ({env.action_space.n} ações)")
    print(f"  - Total de Parâmetros: {sum(p.numel() for p in agent.parameters()):,} tensores")
    print("-" * 75)

    # Treinador Recurrent PPO
    tracker = LiveStatsTracker()
    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        learning_rate=lr,
        rollout_steps=512,
        chunk_length=16,
        chunk_batch_size=16,
        entropy_coef=0.01,
        tracker=tracker,
        device=torch.device("cpu"),
    )

    t0 = time.time()
    final_return = trainer.train(
        max_steps=steps,
        target_return=25.0,
        verbose=True,
    )
    t_total = time.time() - t0

    print("-" * 75)
    print(f"⏱️ Treinamento finalizado em {t_total:.1f}s ({trainer.total_steps} passos coletados)")
    print(f"🎯 Média Móvel de Recompensa Final: {final_return:.2f}")

    # Salva o checkpoint
    KnowledgeTransferManager.save_checkpoint(
        agent=agent,
        checkpoint_path=save_file,
        extra_info={
            "env_id": env_id,
            "mode": mode,
            "scramble_depth": scramble_depth,
            "final_return": final_return,
            "steps": trainer.total_steps,
        },
    )
    print(f"💾 Checkpoint persistido com sucesso: {save_file}")

    # Avaliação com Confidence Gating
    if eval_episodes > 0:
        print("\n" + "=" * 75)
        print(f"🔍 AVALIAÇÃO DE INFERÊNCIA AMORTIZADA ({eval_episodes} EPISÓDIOS)")
        print("=" * 75)
        agent.eval()

        for ep in range(1, eval_episodes + 1):
            obs_dict, info = env.reset()
            agent.reset_memory()
            ep_reward = 0.0
            done = False
            step_count = 0

            print(f"\n--- Episódio {ep}/{eval_episodes} (Inicial: {info['aligned_stickers']}/54 facetas alinhadas) ---")

            while not done and step_count < 20:
                t_inf0 = time.perf_counter_ns()
                decision = agent.act_fast(obs_dict, return_decision=True)
                inf_latency_us = (time.perf_counter_ns() - t_inf0) / 1000.0

                action = decision.action
                action_name = (
                    RubiksCubeCore.MACRO_NAMES[action]
                    if mode == "macro"
                    else RubiksCubeCore.ATOMIC_MOVES[action]
                )

                obs_dict, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                step_count += 1

                conf_pct = decision.confidence * 100.0
                unc_pct = decision.uncertainty * 100.0
                solved_mark = "🏆 RESOLVIDO!" if step_info.get("is_solved") else ""
                print(
                    f"  Passo {step_count:02d} | Ação: {action_name:<16} | "
                    f"Conf: {conf_pct:5.1f}% | Incerteza: {unc_pct:5.1f}% | "
                    f"Latência: {inf_latency_us:5.1f} µs | Alinhadas: {step_info['aligned_stickers']}/54 {solved_mark}"
                )

            print(f"  Resultado Episódio {ep}: Retorno={ep_reward:.2f} | Passos={step_count}")

    print("\n" + "=" * 75)
    print("✅ Treinamento e avaliação do Cubo Mágico concluídos com sucesso!")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Treinamento do System 1 no Cubo Mágico (Atômico & Macro)")
    parser.add_argument(
        "--mode",
        choices=["macro", "atomic"],
        default="macro",
        help="Modo de treinamento: 'macro' (movimentos pré-definidos/CFOP) ou 'atomic' (giros primitivos)",
    )
    parser.add_argument("--steps", type=int, default=10000, help="Passos de treinamento PPO")
    parser.add_argument("--scramble-depth", type=int, default=2, help="Profundidade do embaralhamento no reset")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taxa de aprendizado do PPO")
    parser.add_argument("--save", type=str, default=None, help="Caminho do checkpoint .pt de saída")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Episódios de avaliação ao finalizar")
    args = parser.parse_args()

    train_rubiks(
        mode=args.mode,
        steps=args.steps,
        scramble_depth=args.scramble_depth,
        lr=args.lr,
        save_path=args.save,
        eval_episodes=args.eval_episodes,
    )


if __name__ == "__main__":
    main()
