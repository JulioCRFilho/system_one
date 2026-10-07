#!/usr/bin/env python3
"""Task 14: Treinamento do System 1 no FrozenLake-v1 com Curriculum Learning Reverso.

Demonstra:
1. Reverse Goal Proximity Curriculum:
   - Nível 1: Spawn a 1 passo do Goal (tiles 14, 10).
   - Nível 2: Spawn a 2 passos do Goal (tiles 13, 9, 6).
   - Nível 3: Spawn a 3 passos do Goal (tiles 8, 4, 2).
   - Nível 4: Spawn oficial no início (tile 0).
2. Treinamento de reflexo amortizado com Recurrent PPO sobre observação discreta (16 dims one-hot).
3. Avaliação multinível com telemetria de taxa de resolução, incerteza e latência.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import List, Optional

# Garante import do system1_engine de qualquer diretório
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.adapters.frozenlake import FrozenLakeCurriculumWrapper
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.telemetry.tracker import LiveStatsTracker
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.transfer.manager import KnowledgeTransferManager


def train_frozenlake(
    steps: int = 20000,
    curriculum: bool = True,
    target_success_rate: float = 0.85,
    is_slippery: bool = False,
    lr: float = 2e-3,
    save_path: Optional[str] = None,
    eval_episodes: int = 10,
) -> None:
    env_id = "FrozenLake-v1"
    default_save = "s1_frozenlake_v1_trained.pt"
    save_file = save_path or default_save

    print("=" * 80)
    print(f"❄️ TREINAMENTO DO SYSTEM 1 NO FROZENLAKE-V1 (CURRICULUM LEARNING REVERSO)")
    print(f"Ambiente: {env_id} | Passos Máximos: {steps} | Gelo Escorregadio: {is_slippery}")
    if curriculum:
        print(f"🎓 Curriculum Learning: HABILITADO (Nível 1 -> 4 com meta de {target_success_rate*100:.0f}%)")
    else:
        print("Profundidade Fixa: Nível 4 (Spawn no Início Oficial)")
    print("=" * 80)

    raw_env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
    curr_env = FrozenLakeCurriculumWrapper(
        raw_env,
        curriculum=curriculum,
        target_success_rate=target_success_rate,
    )
    env = UniversalS1Wrapper(curr_env)

    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
    )

    print("• Configuração do Agente Universal S1:")
    print(f"  - Espaço de Observação: VectorFrontEnd ({env.observation_space['obs'].shape[0]} dims one-hot)")
    print(f"  - Tronco Recorrente: GRU + 2x ResMLP (256 latente)")
    print(f"  - Espaço de Ação: CategoricalPolicyHead (4 ações: Left, Down, Right, Up)")
    print(f"  - Total de Parâmetros: {sum(p.numel() for p in agent.parameters()):,} tensores")
    print("-" * 80)

    tracker = LiveStatsTracker()
    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        learning_rate=lr,
        rollout_steps=256,
        chunk_length=16,
        chunk_batch_size=16,
        entropy_coef=0.02,
        tracker=tracker,
        device=torch.device("cpu"),
    )

    t0 = time.time()
    final_return = trainer.train(
        max_steps=steps,
        target_return=1.95,
        verbose=True,
    )
    t_total = time.time() - t0

    achieved_level = curr_env.current_level if curriculum else 4

    print("-" * 80)
    print(f"⏱️ Treinamento finalizado em {t_total:.1f}s ({trainer.total_steps} passos coletados)")
    print(f"🎯 Média Móvel de Recompensa Final: {final_return:.2f}")
    if curriculum:
        print(f"🏆 Nível Alcançado no Curriculum: Nível {achieved_level}/4")

    KnowledgeTransferManager.save_checkpoint(
        agent=agent,
        checkpoint_path=save_file,
        extra_info={
            "env_id": env_id,
            "curriculum": curriculum,
            "achieved_level": achieved_level,
            "is_slippery": is_slippery,
            "final_return": final_return,
            "steps": trainer.total_steps,
        },
    )
    print(f"💾 Checkpoint persistido com sucesso: {save_file}")

    # Avaliação Multinível ao final
    if eval_episodes > 0:
        print("\n" + "=" * 80)
        print(f"🔍 AVALIAÇÃO DE INFERÊNCIA MULTINÍVEL ({eval_episodes} episódios por nível)")
        print("=" * 80)
        agent.eval()

        results_summary = []
        for lvl in range(1, 5):
            solved_count = 0
            total_steps = 0
            total_unc = 0.0
            total_conf = 0.0
            total_lat = 0.0

            for _ in range(eval_episodes):
                obs_dict, info = env.reset(options={"level": lvl})
                agent.reset_memory()
                done = False
                steps_taken = 0

                while not done and steps_taken < 25:
                    t_inf0 = time.perf_counter_ns()
                    decision = agent.act_fast(obs_dict, return_decision=True)
                    inf_lat = (time.perf_counter_ns() - t_inf0) / 1000.0

                    obs_dict, reward, terminated, truncated, step_info = env.step(decision.action)
                    done = terminated or truncated
                    steps_taken += 1

                    total_unc += decision.uncertainty
                    total_conf += decision.confidence
                    total_lat += inf_lat

                    if step_info.get("is_goal"):
                        solved_count += 1
                        break

                total_steps += steps_taken

            solve_pct = (solved_count / eval_episodes) * 100.0
            avg_steps = total_steps / eval_episodes
            avg_unc = (total_unc / max(1, total_steps)) * 100.0
            avg_conf = (total_conf / max(1, total_steps)) * 100.0
            avg_lat = total_lat / max(1, total_steps)

            descriptions = {
                1: "1 passo do Goal (tiles 14, 10)",
                2: "2 passos do Goal (tiles 13, 9, 6)",
                3: "3 passos do Goal (tiles 8, 4, 2)",
                4: "Início oficial (tile 0,0)",
            }

            results_summary.append({
                "level": lvl,
                "desc": descriptions.get(lvl, ""),
                "solved": solved_count,
                "total": eval_episodes,
                "solve_pct": solve_pct,
                "avg_steps": avg_steps,
                "avg_conf": avg_conf,
                "avg_unc": avg_unc,
                "avg_lat": avg_lat,
            })

        print(f"{'Nível':<7} | {'Configuração':<32} | {'Taxa (%)':<10} | {'Passos':<8} | {'Confiança':<10} | {'Incerteza':<10} | {'Latência':<10}")
        print("-" * 80)
        for r in results_summary:
            print(
                f"Nível {r['level']:<1} | {r['desc']:<32} | {r['solve_pct']:>6.1f}%   | "
                f"{r['avg_steps']:>6.1f} | {r['avg_conf']:>6.1f}%    | {r['avg_unc']:>6.1f}%    | {r['avg_lat']:>6.1f} µs"
            )

    print("\n" + "=" * 80)
    print("✅ Treinamento e avaliação do FrozenLake concluídos com sucesso!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Treinamento do System 1 no FrozenLake com Curriculum Learning")
    parser.add_argument("--steps", type=int, default=15000, help="Passos máximos de treinamento")
    parser.add_argument(
        "--curriculum",
        action="store_true",
        default=True,
        help="Habilita curriculum reverso de proximidade ao objetivo",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_false",
        dest="curriculum",
        help="Desabilita curriculum e treina a partir do início oficial",
    )
    parser.add_argument(
        "--target-success-rate",
        type=float,
        default=0.85,
        help="Taxa móvel necessária para promover o nível",
    )
    parser.add_argument(
        "--slippery",
        action="store_true",
        default=False,
        help="Ativa dinâmica estocástica de gelo escorregadio",
    )
    parser.add_argument("--lr", type=float, default=2e-3, help="Taxa de aprendizado")
    parser.add_argument("--save", type=str, default=None, help="Caminho do arquivo .pt de saída")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episódios de avaliação por nível")
    args = parser.parse_args()

    train_frozenlake(
        steps=args.steps,
        curriculum=args.curriculum,
        target_success_rate=args.target_success_rate,
        is_slippery=args.slippery,
        lr=args.lr,
        save_path=args.save,
        eval_episodes=args.eval_episodes,
    )


if __name__ == "__main__":
    main()
