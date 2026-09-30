import gymnasium as gym
import numpy as np
import pytest
import torch

from system1_engine.core.agent import ReflexDecision, UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper


def test_confidence_gating_discrete():
    """Valida o cálculo de incerteza e gatilho do System 2 em espaço discreto."""
    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)

    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    agent.eval()

    obs_dict, _ = env.reset(seed=42)

    # 1. Invocação padrão act_fast (retrocompatibilidade)
    action = agent.act_fast(obs_dict)
    assert isinstance(action, int)
    assert action in [0, 1]

    # 2. Invocação com Confidence Gating
    decision = agent.act_with_confidence(
        obs_dict,
        uncertainty_threshold=0.70,
        confidence_threshold=0.60,
        return_value=True,
    )

    assert isinstance(decision, ReflexDecision)
    assert isinstance(decision.action, int)
    assert 0.0 <= decision.confidence <= 1.0
    assert 0.0 <= decision.uncertainty <= 1.0
    assert 0.0 <= decision.margin <= 1.0
    assert isinstance(decision.is_uncertain, bool)
    assert decision.entropy >= 0.0
    assert decision.latent_value is not None


def test_confidence_gating_continuous():
    """Valida o cálculo de incerteza e gatilho do System 2 em espaço contínuo (Box)."""
    raw_env = gym.make("Pendulum-v1")
    env = UniversalS1Wrapper(raw_env)

    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    agent.eval()

    obs_dict, _ = env.reset(seed=42)

    decision = agent.act_with_confidence(
        obs_dict,
        uncertainty_threshold=0.80,
        return_value=True,
    )

    assert isinstance(decision, ReflexDecision)
    assert isinstance(decision.action, np.ndarray)
    assert decision.action.shape == (1,)
    assert 0.0 <= decision.confidence <= 1.0
    assert 0.0 <= decision.uncertainty <= 1.0
    assert isinstance(decision.is_uncertain, bool)
    assert decision.latent_value is not None


def test_act_fast_latency_with_confidence():
    """Garante que o cálculo de entropia e incerteza não quebra o orçamento de 0.8 ms."""
    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)

    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    agent.eval()

    obs_dict, _ = env.reset(seed=42)

    # Warmup
    for _ in range(50):
        agent.act_with_confidence(obs_dict)

    import time
    latencies = []
    for _ in range(500):
        t0 = time.perf_counter_ns()
        agent.act_with_confidence(obs_dict)
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1e6)

    avg_ms = float(np.mean(latencies))
    assert avg_ms <= 0.80, f"Latência de {avg_ms:.4f} ms excedeu orçamento de 0.8 ms"
