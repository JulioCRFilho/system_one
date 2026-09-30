#!/usr/bin/env python3
"""Task 2: Avaliação e Execução de Checkpoint Treinado (CartPole-v1).

Carrega os pesos consolidados de 's1_cartpole.pt' e executa o agente
no ambiente CartPole-v1 em modo puramente reflexivo com act_fast().
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


def evaluate(
    checkpoint_path: str = "s1_cartpole.pt",
    episodes: int = 5,
    seed: int = 123,
    render: bool = False,
    fps: float = 50.0,
) -> None:
    print("=" * 70)
    print("🎮 TASK 2: AVALIAÇÃO DE CHECKPOINT TREINADO (CartPole-v1)")
    print("=" * 70)

    # 1. Criação do Ambiente Envelopado
    render_mode = "human" if render else None
    raw_env = gym.make("CartPole-v1", render_mode=render_mode)
    env = UniversalS1Wrapper(raw_env)

    # 2. Inicialização do Agente
    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
    )

    # 3. Carregamento dos Pesos
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"❌ Checkpoint '{checkpoint_path}' não encontrado!")
        print("   Execute a Task 3 para treinar um modelo do zero ou verifique o caminho.")
        sys.exit(1)

    print(f"• Carregando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    agent.load_state_dict(state_dict, strict=False)
    agent.eval()

    if isinstance(checkpoint, dict) and "extra_info" in checkpoint:
        info = checkpoint["extra_info"]
        print(f"  - Metadados do Treino: Retorno Final={info.get('final_return', 'N/A')}, Passos={info.get('steps', 'N/A')}")

    print("-" * 70)
    print(f"Executando {episodes} episódios de teste com decisões reflexivas (act_fast)...")
    print("-" * 70)

    episode_rewards = []
    episode_steps = []
    all_latencies_us = []
    dt_target = (1.0 / fps) if (render and fps > 0) else 0.0

    try:
        for ep in range(episodes):
            ep_seed = seed + ep
            obs_dict, _ = env.reset(seed=ep_seed)
            agent.reset_memory()  # Zera memória latente na virada de episódio

            ep_reward = 0.0
            steps = 0
            done = False
            t_start = time.perf_counter()

            while not done:
                t_step_start = time.perf_counter()
                t0 = time.perf_counter_ns()
                action = agent.act_fast(obs_dict)
                t1 = time.perf_counter_ns()
                all_latencies_us.append((t1 - t0) / 1000.0)

                obs_dict, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                steps += 1
                done = terminated or truncated

                if dt_target > 0:
                    elapsed = time.perf_counter() - t_step_start
                    sleep_time = dt_target - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            ep_duration = (time.perf_counter() - t_start) * 1000.0
            episode_rewards.append(ep_reward)
            episode_steps.append(steps)

            status_tag = "🏆 SUCESSO (MAX)" if ep_reward >= 500.0 else "✅ ESTÁVEL"
            print(
                f"Episódio {ep + 1:2d}/{episodes} | "
                f"Recompensa: {ep_reward:5.1f} | "
                f"Passos: {steps:3d} | "
                f"Tempo total: {ep_duration:5.1f} ms | "
                f"{status_tag}"
            )

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        avg_latency = float(np.mean(all_latencies_us))

        print("-" * 70)
        print("📈 RESUMO DA AVALIAÇÃO:")
        print(f"  - Recompensa Média : {mean_reward:.2f} ± {std_reward:.2f} (Máximo teórico: 500.0)")
        print(f"  - Passos Médios    : {np.mean(episode_steps):.1f}")
        print(f"  - Latência Média   : {avg_latency:.1f} µs por passo ({avg_latency / 1000.0:.4f} ms)")
        print("-" * 70)

        if mean_reward >= 475.0:
            print("🌟 [PERFEITO] Agente System 1 dominou o ambiente com controle balanceado!")
        else:
            print("⚠️ [ATENÇÃO] Recompensa abaixo do limiar ótimo de 475.0.")

    finally:
        env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 2: Avaliação de Checkpoint Treinado")
    parser.add_argument("--checkpoint", type=str, default="s1_cartpole.pt", help="Caminho do checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Número de episódios")
    parser.add_argument("--render", action="store_true", default=False, help="Habilita visualização gráfica em tempo real na tela")
    parser.add_argument("--fps", type=float, default=50.0, help="Cadência em FPS ao renderizar (padrão: 50.0)")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        render=args.render,
        fps=args.fps,
    )
