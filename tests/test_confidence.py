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


def test_confidence_gating_continuous_low_sigma():
    """Garante que quando sigma < 0.242 (entropia diferencial negativa), não há underflow de incerteza."""
    raw_env = gym.make("Pendulum-v1")
    env = UniversalS1Wrapper(raw_env)

    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    agent.eval()

    # Força sigma = 0.05 (muito abaixo de 1/sqrt(2*pi*e) ~ 0.24197)
    with torch.no_grad():
        agent.policy_head.log_std.fill_(float(np.log(0.05)))

    obs_dict, _ = env.reset(seed=42)

    decision = agent.act_with_confidence(
        obs_dict,
        uncertainty_threshold=0.70,
        confidence_threshold=0.50,
        return_value=True,
    )

    # 1. A entropia diferencial deve ser estritamente negativa
    assert decision.entropy < 0.0, f"Esperado H < 0 para sigma=0.05, obtido {decision.entropy}"

    # 2. A incerteza deve ser estritamente positiva, limitada em [0, 1] e próxima de 0 (sem underflow)
    assert 0.0 <= decision.uncertainty <= 0.01, f"Incerteza distorcida: {decision.uncertainty}"

    # 3. A confiança deve ser próxima de 100%
    assert decision.confidence >= 0.99, f"Confiança insuficiente para política precisa: {decision.confidence}"

    # 4. Gatilho System 2 deve permanecer DESATIVADO
    assert decision.is_uncertain is False, "Gatilho System 2 ativado indevidamente para política altamente certa"


def test_confidence_gating_continuous_extreme_small_sigma():
    """Garante robustez numérica mesmo quando log_std é extremamente negativo (-15.0)."""
    raw_env = gym.make("Pendulum-v1")
    env = UniversalS1Wrapper(raw_env)

    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    agent.eval()

    # Força log_std = -15.0 (sigma ~ 3e-7)
    with torch.no_grad():
        agent.policy_head.log_std.fill_(-15.0)

    obs_dict, _ = env.reset(seed=42)
    decision = agent.act_with_confidence(obs_dict)

    assert decision.entropy < -10.0
    assert 0.0 <= decision.uncertainty <= 1e-4
    assert decision.confidence >= 0.9999
    assert decision.is_uncertain is False


def test_confidence_gating_continuous_high_sigma():
    """Garante ativação correta do gatilho quando a política é altamente incerta (sigma = 2.0)."""
    raw_env = gym.make("Pendulum-v1")
    env = UniversalS1Wrapper(raw_env)

    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    agent.eval()

    with torch.no_grad():
        agent.policy_head.log_std.fill_(float(np.log(2.0)))

    obs_dict, _ = env.reset(seed=42)
    decision = agent.act_with_confidence(
        obs_dict,
        uncertainty_threshold=0.70,
        confidence_threshold=0.50,
    )

    assert decision.entropy > 2.0
    assert decision.uncertainty >= 0.95
    assert decision.confidence <= 0.05
    assert decision.is_uncertain is True


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
