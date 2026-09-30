from typing import Optional
import argparse
import sys
import time
import gymnasium as gym
import numpy as np
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.transfer.manager import KnowledgeTransferManager


def train_mode(args: argparse.Namespace) -> None:
    print(f"=== Starting Training Mode on {args.env} ===")
    raw_env = gym.make(args.env)
    env = UniversalS1Wrapper(raw_env)

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

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        learning_rate=args.lr,
        rollout_steps=args.rollout_steps,
        chunk_length=args.chunk_length,
        chunk_batch_size=args.chunk_batch_size,
    )

    final_return = trainer.train(
        max_steps=args.steps,
        target_return=args.target_return,
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


def run_mode(args: argparse.Namespace) -> None:
    print(f"=== Running Agent Evaluation on {args.env} ===")
    raw_env = gym.make(args.env)
    env = UniversalS1Wrapper(raw_env)

    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
    )

    if args.load:
        checkpoint = torch.load(args.load, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        agent.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {args.load}")

    agent.eval()
    for ep in range(args.episodes):
        obs_dict, _ = env.reset()
        agent.reset_memory()
        ep_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.act_fast(obs_dict)
            obs_dict, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"Episode {ep + 1}/{args.episodes} | Return: {ep_reward:.1f} | Steps: {steps}")


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
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment ID")
    parser.add_argument("--steps", type=int, default=40000, help="Training steps or benchmark steps")
    parser.add_argument("--save", type=str, default="s1_cartpole.pt", help="Path to save checkpoint")
    parser.add_argument("--load", type=str, default=None, help="Path to load checkpoint for run mode")
    parser.add_argument(
        "--transfer-from",
        type=str,
        default=None,
        help="Path to load transferable trunk weights for cross-scenario transfer",
    )
    parser.add_argument(
        "--freeze-trunk",
        action="store_true",
        default=True,
        help="Freeze trunk parameters during transfer",
    )
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

    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args)
    elif args.mode == "run":
        run_mode(args)
    elif args.mode == "benchmark":
        benchmark_mode(args)


if __name__ == "__main__":
    main()
