import argparse
import sys
import time
from typing import Optional
import gymnasium as gym
import numpy as np
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.adapters import make_game_env
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.telemetry import LiveStatsTracker, S1LiveDashboard, TelemetryServer
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.transfer.manager import KnowledgeTransferManager


def build_environment(
    env_id: str,
    scenario: Optional[str] = None,
    render: bool = False,
) -> UniversalS1Wrapper:
    """Instancia o ambiente apropriado (Gymnasium padrão ou Adaptador Nativo ViZDoom)."""
    if env_id.lower() in ["vizdoom", "native"]:
        scenario_path = scenario or "basic.cfg"
        return make_game_env(
            "native",
            {
                "engine_type": "vizdoom",
                "scenario_path": scenario_path,
                "args": {"headless": not render},
            },
        )
    render_mode = "human" if render else None
    raw_env = gym.make(env_id, render_mode=render_mode)
    return UniversalS1Wrapper(raw_env)


def train_mode(args: argparse.Namespace) -> None:
    print(f"=== Starting Training Mode on {args.env} ===")
    env = build_environment(args.env, scenario=args.scenario, render=getattr(args, "render", False))

    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
    )

    if args.transfer_from:
        print(f"Transferring trunk weights from: {args.transfer_from}")
        loaded = KnowledgeTransferManager.load_transferable_weights(
            agent=agent,
            checkpoint_path=args.transfer_from,
            freeze_trunk=args.freeze_trunk,
        )
        print(f"Loaded {len(loaded)} trunk parameters. Trunk frozen: {args.freeze_trunk}")

    need_tracker = args.live_stats or args.web_panel
    tracker = LiveStatsTracker() if need_tracker else None
    dashboard = S1LiveDashboard(tracker) if args.live_stats else None
    should_open_browser = args.web_panel and not getattr(args, "no_browser", False) and sys.stdin.isatty()
    web_server = (
        TelemetryServer(tracker, host=args.host, port=args.port, open_browser=should_open_browser)
        if args.web_panel and tracker
        else None
    )

    if web_server:
        web_server.start()

    try:
        trainer = RecurrentPPOTrainer(
            agent=agent,
            env=env,
            learning_rate=args.lr,
            rollout_steps=args.rollout_steps,
            chunk_length=args.chunk_length,
            chunk_batch_size=args.chunk_batch_size,
            entropy_coef=args.entropy_coef,
            tracker=tracker,
        )

        def train_callback(steps: int, mean_ret: float) -> None:
            if dashboard:
                dashboard.update()

        if dashboard:
            with dashboard:
                final_return = trainer.train(
                    max_steps=args.steps,
                    target_return=args.target_return,
                    callback=train_callback,
                    verbose=False,
                )
            dashboard.render_once()
        else:
            final_return = trainer.train(
                max_steps=args.steps,
                target_return=args.target_return,
                callback=train_callback,
                verbose=True,
            )

        if args.save:
            KnowledgeTransferManager.save_checkpoint(
                agent=agent,
                checkpoint_path=args.save,
                extra_info={
                    "env_id": args.env,
                    "final_return": final_return,
                    "steps": trainer.total_steps,
                },
            )
            print(f"Checkpoint successfully saved to: {args.save}")

        if tracker:
            tracker.set_completed(True)

        if web_server:
            dashboard_url = f"http://{args.host}:{args.port}"
            print("\n" + "=" * 70)
            print("🏁 Treinamento concluído com sucesso!")
            print(f"🌐 Painel Web mantido ativo para inspeção em: {dashboard_url}")
            print("⌨️  Pressione Ctrl+C no terminal para encerrar o servidor...")
            print("=" * 70)
            if not getattr(args, "no_wait", False) and sys.stdin.isatty():
                try:
                    while True:
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    print("\nEncerrando servidor web...")

    finally:
        if web_server:
            web_server.stop()
        env.close()


def run_mode(args: argparse.Namespace) -> None:
    print(f"=== Running Agent Evaluation on {args.env} ===")
    env = build_environment(args.env, scenario=args.scenario, render=getattr(args, "render", False))

    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
    )

    if args.load:
        checkpoint = torch.load(args.load, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        
        compat_err = KnowledgeTransferManager.validate_evaluation_compatibility(agent, state_dict, args.env, args.load)
        if compat_err:
            print(compat_err)
            sys.exit(1)

        agent.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {args.load}")

    agent.eval()
    need_tracker = args.live_stats or args.web_panel
    tracker = LiveStatsTracker() if need_tracker else None
    dashboard = S1LiveDashboard(tracker) if args.live_stats else None
    should_open_browser = args.web_panel and not getattr(args, "no_browser", False) and sys.stdin.isatty()
    web_server = (
        TelemetryServer(tracker, host=args.host, port=args.port, open_browser=should_open_browser)
        if args.web_panel and tracker
        else None
    )

    if web_server:
        web_server.start()

    target_fps = args.fps
    if target_fps is None:
        target_fps = 50.0 if (args.web_panel or getattr(args, "render", False)) else 0.0

    dt_target = (1.0 / target_fps) if target_fps > 0 else 0.0

    try:
        def execute_eval_loop() -> None:
            for ep in range(args.episodes):
                obs_dict, _ = env.reset()
                agent.reset_memory()
                ep_reward = 0.0
                steps = 0
                done = False

                while not done:
                    t_step_start = time.perf_counter()
                    if tracker is not None:
                        t0 = time.perf_counter_ns()
                        decision = agent.act_with_confidence(obs_dict)
                        lat_us = (time.perf_counter_ns() - t0) / 1000.0

                        tracker.record_inference(
                            latency_us=lat_us,
                            uncertainty=decision.uncertainty,
                            confidence=decision.confidence,
                            entropy=decision.entropy,
                        )
                        action = decision.action
                    else:
                        action = agent.act_fast(obs_dict)

                    obs_dict, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    if tracker is not None:
                        tracker.record_env_step(reward=reward, done=done)
                        if dashboard:
                            dashboard.update()

                    ep_reward += reward
                    steps += 1

                    if dt_target > 0:
                        elapsed = time.perf_counter() - t_step_start
                        sleep_time = dt_target - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                if not args.live_stats:
                    print(f"Episode {ep + 1}/{args.episodes} | Return: {ep_reward:.1f} | Steps: {steps}")

        if dashboard:
            with dashboard:
                execute_eval_loop()
            dashboard.render_once()
        else:
            execute_eval_loop()

        if tracker:
            tracker.set_completed(True)

        if web_server:
            dashboard_url = f"http://{args.host}:{args.port}"
            print("\n" + "=" * 70)
            print(f"🏁 Avaliação concluída com sucesso! ({args.episodes} episódios)")
            print(f"🌐 Painel Web mantido ativo para inspeção em: {dashboard_url}")
            print("⌨️  Pressione Ctrl+C no terminal para encerrar o servidor...")
            print("=" * 70)
            if not getattr(args, "no_wait", False) and sys.stdin.isatty():
                try:
                    while True:
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    print("\nEncerrando servidor web...")

    finally:
        if web_server:
            web_server.stop()
        env.close()


def benchmark_mode(args: argparse.Namespace) -> None:
    print("=== Running act_fast() CPU Latency Benchmark ===")
    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)

    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
    )
    agent.eval()

    obs_dict, _ = env.reset()
    agent.reset_memory()

    # Warmup
    for _ in range(50):
        agent.act_fast(obs_dict)

    latencies = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        agent.act_fast(obs_dict)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    avg_latency = float(np.mean(latencies))
    med_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    p99_latency = float(np.percentile(latencies, 99))
    min_latency = float(np.min(latencies))
    max_latency = float(np.max(latencies))

    print(f"Benchmark over {args.steps} sequential steps:")
    print(f"  Average Latency : {avg_latency:.4f} ms (Budget: <= 0.8000 ms)")
    print(f"  Median Latency  : {med_latency:.4f} ms")
    print(f"  P95 Latency     : {p95_latency:.4f} ms")
    print(f"  P99 Latency     : {p99_latency:.4f} ms")
    print(f"  Min / Max       : {min_latency:.4f} ms / {max_latency:.4f} ms")

    if avg_latency <= 0.8:
        print("[PASS] Latency budget satisfied (<= 0.8 ms).")
    else:
        print(f"[FAIL] Latency budget exceeded: {avg_latency:.4f} ms > 0.8 ms")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal System 1 RL Agent Engine")
    parser.add_argument(
        "--mode",
        choices=["train", "run", "benchmark"],
        default="benchmark",
        help="Operating mode: train, run, or benchmark",
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment ID or 'vizdoom'")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario file for vizdoom/native environment")
    parser.add_argument("--steps", type=int, default=40000, help="Training steps or benchmark steps")
    parser.add_argument("--save", type=str, default=None, help="Path to save checkpoint (default: None)")
    parser.add_argument("--load", type=str, default=None, help="Path to load checkpoint for run mode")
    parser.add_argument(
        "--transfer-from",
        type=str,
        default=None,
        help="Path to load transferable trunk weights for cross-scenario transfer",
    )
    parser.add_argument(
        "--freeze-trunk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze trunk parameters during transfer (use --no-freeze-trunk to fine-tune the trunk)",
    )
    parser.add_argument("--entropy-coef", type=float, default=0.005, help="Entropy coefficient for PPO exploration (default: 0.005)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes for run mode")
    parser.add_argument("--lr", type=float, default=7e-4, help="Learning rate")
    parser.add_argument(
        "--target-return",
        type=float,
        default=475.0,
        help="Target moving average return to stop training early",
    )
    parser.add_argument("--rollout-steps", type=int, default=1024, help="Rollout steps per PPO iteration")
    parser.add_argument("--chunk-length", type=int, default=16, help="Chunk sequence length for BPTT")
    parser.add_argument("--chunk-batch-size", type=int, default=16, help="Chunk batch size")
    parser.add_argument(
        "--live-stats",
        action="store_true",
        default=False,
        help="Enable real-time Rich terminal live telemetry dashboard",
    )
    parser.add_argument(
        "--web-panel",
        action="store_true",
        default=False,
        help="Launch real-time web telemetry dashboard (HTTP/SSE on port 8050)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="HTTP port for web telemetry server (default: 8050)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for web telemetry server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Cadência em FPS para o modo run (padrão: 50.0 quando --web-panel está ativo, ou 0 para velocidade máxima desimpedida)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        default=False,
        help="Não aguarda confirmação com Ctrl+C ao término da execução com --web-panel",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=False,
        help="Não abre o navegador automaticamente ao iniciar com --web-panel",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Habilita visualização gráfica do ambiente em tempo real na tela (render_mode='human')",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args)
    elif args.mode == "run":
        run_mode(args)
    elif args.mode == "benchmark":
        benchmark_mode(args)


if __name__ == "__main__":
    main()
