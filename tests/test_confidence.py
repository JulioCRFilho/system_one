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


def test_act_fast_stochastic_sampling_continuous():
    """Valida que act_fast e act_with_confidence suportam modos determinístico e estocástico contínuo."""
    raw_env = gym.make("Pendulum-v1")
    env = UniversalS1Wrapper(raw_env)

    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    agent.eval()

    obs_dict, _ = env.reset(seed=42)

    # 1. Determinístico: chamadas repetidas produzem exatamente a mesma ação
    agent.reset_memory()
    act_det1 = agent.act_fast(obs_dict, deterministic=True)
    agent.reset_memory()
    act_det2 = agent.act_fast(obs_dict, deterministic=True)
    np.testing.assert_allclose(act_det1, act_det2, rtol=1e-5, atol=1e-5)

    # 2. Estocástico (ruído calibrado noise_scale=0.25): chamadas produzem ações ligeiramente diferentes
    agent.reset_memory()
    act_stoch1 = agent.act_fast(obs_dict, deterministic=False, noise_scale=0.25)
    agent.reset_memory()
    act_stoch2 = agent.act_fast(obs_dict, deterministic=False, noise_scale=0.25)
    assert not np.allclose(act_stoch1, act_stoch2)

    # 3. act_with_confidence com noise_scale
    agent.reset_memory()
    dec_det = agent.act_with_confidence(obs_dict, deterministic=True)
    agent.reset_memory()
    dec_stoch = agent.act_with_confidence(obs_dict, deterministic=False, noise_scale=0.5)
    assert isinstance(dec_det, ReflexDecision)
    assert isinstance(dec_stoch, ReflexDecision)
    assert not np.allclose(dec_det.action, dec_stoch.action)


def test_act_fast_stochastic_sampling_discrete():
    """Valida que act_fast suporta amostragem discreta categórica."""
    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)

    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    agent.eval()

    obs_dict, _ = env.reset(seed=42)

    act_det = agent.act_fast(obs_dict, deterministic=True)
    assert act_det in [0, 1]

    act_stoch = agent.act_fast(obs_dict, deterministic=False)
    assert act_stoch in [0, 1]


def test_calibration_continuous():
    """Valida calibração contínua de 0.0 a 1.0 para ações contínuas."""
    raw_env = gym.make("Pendulum-v1")
    env = UniversalS1Wrapper(raw_env)
    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    agent.eval()
    obs_dict, _ = env.reset(seed=42)

    # 1. Calibração 0.0 é estritamente determinística
    agent.reset_memory()
    act0_a = agent.act_fast(obs_dict, calibration=0.0)
    agent.reset_memory()
    act0_b = agent.act_fast(obs_dict, calibration=0.0)
    np.testing.assert_allclose(act0_a, act0_b, rtol=1e-5, atol=1e-5)

    # 2. Calibração > 0.0 varia estocasticamente proporcional à calibração
    agent.reset_memory()
    act_calib_a = agent.act_fast(obs_dict, calibration=0.5)
    agent.reset_memory()
    act_calib_b = agent.act_fast(obs_dict, calibration=0.5)
    assert not np.allclose(act_calib_a, act_calib_b)

    # 3. act_with_confidence respeita calibration
    agent.reset_memory()
    dec0 = agent.act_with_confidence(obs_dict, calibration=0.0)
    agent.reset_memory()
    dec1 = agent.act_with_confidence(obs_dict, calibration=1.0)
    assert isinstance(dec0, ReflexDecision)
    assert isinstance(dec1, ReflexDecision)
    np.testing.assert_allclose(dec0.action, act0_a, rtol=1e-5, atol=1e-5)


def test_calibration_discrete():
    """Valida calibração contínua de 0.0 a 1.0 para ações discretas."""
    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)
    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    agent.eval()
    obs_dict, _ = env.reset(seed=42)

    # 1. Calibração 0.0 é estritamente argmax
    agent.reset_memory()
    act0_a = agent.act_fast(obs_dict, calibration=0.0)
    agent.reset_memory()
    act0_b = agent.act_fast(obs_dict, calibration=0.0)
    assert act0_a == act0_b

    # 2. Amostragem repetida com calibração intermediária (0.5) e total (1.0)
    actions_calib = []
    for _ in range(30):
        agent.reset_memory()
        actions_calib.append(agent.act_fast(obs_dict, calibration=0.5))
    assert all(a in [0, 1] for a in actions_calib)

    # 3. act_with_confidence com calibration
    agent.reset_memory()
    dec0 = agent.act_with_confidence(obs_dict, calibration=0.0)
    assert dec0.action == act0_a
    assert 0.0 <= dec0.confidence <= 1.0


def test_ppo_exploration_scale_training():
    """Valida que o PPO respeita exploration_scale tanto na coleta de rollouts quanto no treino."""
    from system1_engine.training.ppo import RecurrentPPOTrainer
    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)
    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        rollout_steps=64,
        chunk_length=8,
        chunk_batch_size=4,
        exploration_scale=0.5,
    )
    assert trainer.exploration_scale == 0.5

    obs_dict, _ = env.reset(seed=42)
    obs_dict, hx, ep_start, ret = trainer.collect_rollouts(
        current_obs_dict=obs_dict,
        current_hx=None,
        episode_start=True,
    )
    assert len(trainer.buffer.rewards) == 64

    # Treina uma época com a escala de exploração calibrada
    metrics = trainer.train_epoch()
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert not np.isnan(metrics["policy_loss"])


def test_auto_calibration_homeostasis():
    """Valida a homeostase neuromoduladora do System 1 (auto-calibração termodinâmica).
    - Quando estagnado (recompensa nula/baixa), aquece a calibração para inovar.
    - Quando obtém progresso (recompensa positiva), resfria para foco determinístico.
    """
    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)
    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    agent.eval()

    obs_dict, _ = env.reset(seed=42)
    agent.reset_memory()

    # 1. Estado inicial é calibração mínima (0.0)
    assert agent._adaptive_calibration == 0.0
    assert agent._stagnation_count == 0

    # 2. Passos estagnados (prev_reward = 0.0)
    obs_zero = dict(obs_dict)
    obs_zero["prev_reward"] = np.array([0.0], dtype=np.float32)
    dec1 = agent.act_with_confidence(obs_zero, auto_calibrate=True)
    assert dec1.calibration == 0.0  # Ainda não atingiu o limiar de 2 passos de estagnação

    # Passos seguintes com estagnação aumentam a temperatura adaptativa
    calibs = []
    for _ in range(6):
        dec = agent.act_with_confidence(obs_zero, auto_calibrate=True)
        calibs.append(dec.calibration)

    # A temperatura deve ter aumentado monotonicamente até o cap (0.80)
    assert calibs[-1] > 0.0, f"Calibração deveria ter aquecido sob estagnação, obtido {calibs[-1]}"
    assert calibs[-1] <= 0.80

    # 3. Alívio de estagnação / Sucesso (prev_reward = +1.0)
    obs_reward = dict(obs_dict)
    obs_reward["prev_reward"] = np.array([1.0], dtype=np.float32)

    dec_relief1 = agent.act_with_confidence(obs_reward, auto_calibrate=True)
    assert agent._stagnation_count == 0
    # Deve ter resfriado
    assert dec_relief1.calibration < calibs[-1]

    # Mais passos com alta recompensa devem resfriar até 0.0
    for _ in range(5):
        dec_relief = agent.act_with_confidence(obs_reward, auto_calibrate=True)

    assert dec_relief.calibration == 0.0, f"Esperado resfriamento a 0.0, obtido {dec_relief.calibration}"

    # 4. Também suporta act_fast(auto_calibrate=True) ou calibration='auto'
    agent.reset_memory()
    act = agent.act_fast(obs_zero, calibration="auto")
    assert act in [0, 1]
    assert 0.0 <= agent._adaptive_calibration <= 0.80


def test_ppo_auto_exploration_training():
    """Valida o modo de exploração automático e homeostase dinâmica no RecurrentPPOTrainer.
    - Suporta string 'auto', 'adaptive' ou 'homeostatic'
    - Ajusta a escala dinamicamente com base em recompensa/estagnação
    - Registra a calibração corrente na telemetria
    - Treina épocas sem divergência ou NaNs
    """
    from system1_engine.training.ppo import RecurrentPPOTrainer
    from system1_engine.telemetry.tracker import LiveStatsTracker

    # 1. Teste em ambiente discreto (CartPole-v1)
    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)
    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    tracker = LiveStatsTracker()

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        rollout_steps=64,
        chunk_length=8,
        chunk_batch_size=4,
        exploration_scale="auto",
        tracker=tracker,
    )
    assert trainer.is_auto_exploration is True
    assert trainer._adaptive_scale == 0.80

    obs_dict, _ = env.reset(seed=42)
    obs_dict, hx, ep_start, ret = trainer.collect_rollouts(
        current_obs_dict=obs_dict,
        current_hx=None,
        episode_start=True,
    )
    assert len(trainer.buffer.rewards) == 64
    assert 0.15 <= trainer.exploration_scale <= 1.0

    # Verifica telemetria populada com calibração
    snapshot = tracker.snapshot()
    assert "calibration" in snapshot
    assert snapshot["calibration"] > 0.0

    # Executa otimização de gradiente
    metrics = trainer.train_epoch()
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert not np.isnan(metrics["policy_loss"])
    assert not np.isnan(metrics["value_loss"])

    # 2. Teste em ambiente contínuo (Pendulum-v1)
    raw_cont_env = gym.make("Pendulum-v1")
    cont_env = UniversalS1Wrapper(raw_cont_env)
    cont_agent = UniversalS1Agent(obs_space=cont_env.observation_space, action_space=cont_env.action_space)

    cont_trainer = RecurrentPPOTrainer(
        agent=cont_agent,
        env=cont_env,
        rollout_steps=64,
        chunk_length=8,
        chunk_batch_size=4,
        exploration_scale="auto",
    )
    assert cont_trainer.is_auto_exploration is True

    cont_obs, _ = cont_env.reset(seed=42)
    cont_obs, _, _, _ = cont_trainer.collect_rollouts(
        current_obs_dict=cont_obs,
        current_hx=None,
        episode_start=True,
    )
    assert len(cont_trainer.buffer.rewards) == 64
    cont_metrics = cont_trainer.train_epoch()
    assert not np.isnan(cont_metrics["policy_loss"])
    assert not np.isnan(cont_metrics["value_loss"])




