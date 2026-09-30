import time
import gymnasium as gym
import numpy as np
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper


def test_act_fast_latency_budget():
    """Verify agent.act_fast() executes in <= 0.8 ms on CPU across 1,000 steps."""
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
    num_steps = 1000

    for _ in range(num_steps):
        t0 = time.perf_counter()
        action = agent.act_fast(obs_dict)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # Convert to ms

        # Verify action type
        assert isinstance(action, int), f"Discrete act_fast must return int, got {type(action)}"

    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    max_latency = float(np.max(latencies))

    print(
        f"\nCPU Latency Benchmark ({num_steps} steps): "
        f"Mean: {avg_latency:.4f} ms | P95: {p95_latency:.4f} ms | Max: {max_latency:.4f} ms"
    )

    # Hard acceptance criterion: <= 0.8 ms average
    assert avg_latency <= 0.8, (
        f"Latency budget violated: average {avg_latency:.4f} ms > 0.8 ms"
    )


def test_act_fast_continuous_action():
    """Verify act_fast returns numpy array for continuous action spaces."""
    raw_env = gym.make("Pendulum-v1")
    env = UniversalS1Wrapper(raw_env)

    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
    )
    agent.eval()

    obs_dict, _ = env.reset()
    action = agent.act_fast(obs_dict)

    assert isinstance(action, np.ndarray), f"Continuous act_fast must return np.ndarray, got {type(action)}"
    assert action.shape == env.action_space.shape
