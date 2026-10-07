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
    print("\n" + "=" * 80)
    print("🧩 CATÁLOGO DE MACRO-AÇÕES SIMÉTRICAS DO SYSTEM 1 (PARES INVERSOS EXATOS)")
    print("=" * 80)
    print(f"{'Ação':<4} | {'Nome da Macro':<20} | {'Sequência de Giros':<28} | {'Efeito Controlado'}")
    print("-" * 80)
    descriptions = {
        "SEXY_MOVE_R": "Trigger Direito (troca/orienta quinas)",
        "SEXY_MOVE_R_PRIME": "Inverso exato do Trigger Direito",
        "SEXY_MOVE_L": "Trigger Esquerdo (simetria)",
        "SEXY_MOVE_L_PRIME": "Inverso exato do Trigger Esquerdo",
        "SUNE": "Orientação de cantos amarelos (topo)",
        "ANTI_SUNE": "Inverso exato do Sune (topo)",
        "YELLOW_CROSS": "Fru-Ruf (cria a cruz amarela no topo)",
        "YELLOW_CROSS_PRIME": "Inverso exato do Fru-Ruf",
        "ROTATE_Y": "Giro do cubo todo no eixo Y (+90°)",
        "ROTATE_Y_PRIME": "Giro do cubo todo no eixo Y (-90°)",
        "U_TURN": "Giro da camada superior (+90°)",
        "U_PRIME_TURN": "Giro inverso da camada superior (-90°)",
    }
    for idx, (name, seq) in enumerate(RubiksCubeCore.MACRO_ACTIONS.items()):
        seq_str = " ".join(seq)
        if len(seq_str) > 26:
            seq_str = seq_str[:23] + "..."
        desc = descriptions.get(name, "Movimento pré-definido")
        print(f"{idx:<4} | {name:<20} | {seq_str:<28} | {desc}")
    print("=" * 80 + "\n")


def train_rubiks(
    mode: str = "atomic",
    steps: int = 40000,
    curriculum: bool = True,
    min_depth: int = 1,
    max_depth: int = 4,
    target_success_rate: float = 0.90,
    scramble_depth: int = 2,
    lr: float = 1e-3,
    save_path: Optional[str] = None,
    eval_episodes: int = 5,
    eval_depths: Optional[List[int]] = None,
) -> None:
    env_id = "RubiksCubeMacro-v0" if mode == "macro" else "RubiksCube-v0"
    default_save = f"s1_{'rubiks_macro' if mode == 'macro' else 'rubikscube_v0'}_trained.pt"
    save_file = save_path or default_save

    print("=" * 80)
    print(f"🎲 TREINAMENTO DO SYSTEM 1 NO CUBO MÁGICO ({mode.upper()})")
    print(f"Ambiente: {env_id} | Passos Máximos: {steps}")
    if curriculum:
        print(f"🎓 Curriculum Learning: HABILITADO (Depth {min_depth} -> {max_depth} com meta de {target_success_rate*100:.0f}% de resolução)")
    else:
        print(f"Profundidade Fixa: {scramble_depth}")
    print("=" * 80)

    if mode == "macro":
        display_macro_catalog()
        raw_env = RubiksCubeMacroEnv(
            scramble_depth=scramble_depth,
            curriculum=curriculum,
            min_depth=min_depth,
            max_depth=max_depth,
            target_success_rate=target_success_rate,
        )
    else:
        raw_env = RubiksCubeEnv(
            scramble_depth=scramble_depth,
            curriculum=curriculum,
            min_depth=min_depth,
            max_depth=max_depth,
            target_success_rate=target_success_rate,
        )

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
    print("-" * 80)

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
        target_return=35.0,
        verbose=True,
    )
    t_total = time.time() - t0

    achieved_depth = raw_env.current_depth if curriculum else scramble_depth

    print("-" * 80)
    print(f"⏱️ Treinamento finalizado em {t_total:.1f}s ({trainer.total_steps} passos coletados)")
    print(f"🎯 Média Móvel de Recompensa Final: {final_return:.2f}")
    if curriculum:
        print(f"🏆 Profundidade Alcançada no Curriculum: Depth {achieved_depth}/{max_depth}")

    # Salva o checkpoint com metadados completos
    KnowledgeTransferManager.save_checkpoint(
        agent=agent,
        checkpoint_path=save_file,
        extra_info={
            "env_id": env_id,
            "mode": mode,
            "curriculum": curriculum,
            "achieved_depth": achieved_depth,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "scramble_depth": scramble_depth,
            "final_return": final_return,
            "steps": trainer.total_steps,
        },
    )
    print(f"💾 Checkpoint persistido com sucesso: {save_file}")

    # Avaliação Multiprofundidade
    if eval_episodes > 0:
        target_eval_depths = eval_depths or list(range(min_depth, (max_depth if curriculum else scramble_depth) + 1))
        print("\n" + "=" * 80)
        print(f"🔍 AVALIAÇÃO DE INFERÊNCIA MULTIPROFUNDIDADE ({eval_episodes} episódios por nível)")
        print("=" * 80)
        agent.eval()

        results_summary = []

        for d in target_eval_depths:
            solved_count = 0
            total_steps = 0
            total_uncertainty = 0.0
            total_confidence = 0.0
            total_latency_us = 0.0
            ep_count = 0

            for ep in range(1, eval_episodes + 1):
                obs_dict, info = env.reset(options={"scramble_depth": d})
                agent.reset_memory()
                done = False
                steps_taken = 0

                while not done and steps_taken < (d * 4 + 10):
                    t_inf0 = time.perf_counter_ns()
                    decision = agent.act_fast(obs_dict, return_decision=True)
                    inf_lat = (time.perf_counter_ns() - t_inf0) / 1000.0

                    action = decision.action
                    obs_dict, reward, terminated, truncated, step_info = env.step(action)
                    done = terminated or truncated
                    steps_taken += 1

                    total_uncertainty += decision.uncertainty
                    total_confidence += decision.confidence
                    total_latency_us += inf_lat

                    if step_info.get("is_solved"):
                        solved_count += 1
                        break

                total_steps += steps_taken
                ep_count += 1

            solve_pct = (solved_count / max(1, ep_count)) * 100.0
            avg_steps = total_steps / max(1, ep_count)
            avg_unc = (total_uncertainty / max(1, total_steps)) * 100.0
            avg_conf = (total_confidence / max(1, total_steps)) * 100.0
            avg_lat = total_latency_us / max(1, total_steps)

            results_summary.append({
                "depth": d,
                "solved": solved_count,
                "total": ep_count,
                "solve_pct": solve_pct,
                "avg_steps": avg_steps,
                "avg_conf": avg_conf,
                "avg_unc": avg_unc,
                "avg_lat": avg_lat,
            })

        print(f"{'Depth':<6} | {'Resoluções':<12} | {'Taxa (%)':<10} | {'Passos Médios':<14} | {'Confiança':<10} | {'Incerteza':<10} | {'Latência':<10}")
        print("-" * 80)
        for r in results_summary:
            print(
                f"Depth {r['depth']:<1} | {r['solved']:>2}/{r['total']:<2} ep      | {r['solve_pct']:>6.1f}%   | "
                f"{r['avg_steps']:>8.1f} passos | {r['avg_conf']:>6.1f}%    | {r['avg_unc']:>6.1f}%    | {r['avg_lat']:>6.1f} µs"
            )

    print("\n" + "=" * 80)
    print("✅ Treinamento e avaliação com Curriculum Learning concluídos!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Treinamento do System 1 no Cubo Mágico com Curriculum Learning")
    parser.add_argument(
        "--mode",
        choices=["macro", "atomic"],
        default="atomic",
        help="Modo de treinamento: 'atomic' (giros primitivos) ou 'macro' (algoritmos pré-definidos simétricos)",
    )
    parser.add_argument("--steps", type=int, default=40000, help="Passos de treinamento PPO (ex: 40000)")
    parser.add_argument(
        "--curriculum",
        action="store_true",
        default=True,
        help="Habilita escalonamento automático de profundidade (Curriculum Learning)",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_false",
        dest="curriculum",
        help="Desabilita curriculum learning e usa scramble-depth fixo",
    )
    parser.add_argument("--min-depth", type=int, default=1, help="Profundidade inicial do curriculum")
    parser.add_argument("--max-depth", type=int, default=4, help="Profundidade máxima almejada no curriculum")
    parser.add_argument(
        "--target-success-rate",
        type=float,
        default=0.90,
        help="Taxa móvel de vitórias necessária para promover a profundidade (ex: 0.90)",
    )
    parser.add_argument("--scramble-depth", type=int, default=2, help="Profundidade fixa quando curriculum desativado")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taxa de aprendizado do PPO")
    parser.add_argument("--save", type=str, default=None, help="Caminho do checkpoint .pt de saída")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episódios de avaliação por nível ao finalizar")
    args = parser.parse_args()

    train_rubiks(
        mode=args.mode,
        steps=args.steps,
        curriculum=args.curriculum,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        target_success_rate=args.target_success_rate,
        scramble_depth=args.scramble_depth,
        lr=args.lr,
        save_path=args.save,
        eval_episodes=args.eval_episodes,
    )


if __name__ == "__main__":
    main()
