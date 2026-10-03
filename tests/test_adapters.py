import time
import gymnasium as gym
import numpy as np
import pytest

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.adapters import (
    BufferMemoryBackend,
    MemoryField,
    MemoryHookEnv,
    NativeEngineEnv,
    WindowCaptureEnv,
    make_game_env,
)
from system1_engine.env.wrapper import UniversalS1Wrapper


def test_window_capture_env_mock_and_agent_pipeline():
    """Testa Nível 1 (WindowCaptureEnv): Frame pacing, formato (C, 84, 84) e inferência act_fast."""
    # Gera frame sintético 200x200x3
    mock_frame = np.full((200, 200, 3), fill_value=128, dtype=np.uint8)

    env = WindowCaptureEnv(
        window_bbox={"top": 0, "left": 0, "width": 200, "height": 200},
        actions_map={0: None, 1: "space"},
        target_fps=60,
        grayscale=True,
        mock_capture_source=lambda: mock_frame,
        reward_fn=lambda curr, prev, info: 1.0,
    )

    # 1. Validação de Espaços
    assert env.observation_space.shape == (1, 84, 84)
    assert env.action_space.n == 2

    # 2. Reset e Step
    obs, info = env.reset()
    assert obs.shape == (1, 84, 84)
    assert obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0

    obs, reward, terminated, truncated, step_info = env.step(1)
    assert reward == 1.0
    assert not terminated
    assert "step_time_ms" in step_info

    # 3. Integração com UniversalS1Wrapper e UniversalS1Agent
    wrapped = UniversalS1Wrapper(env, is_visual=True)
    obs_dict, _ = wrapped.reset()
    assert obs_dict["obs"].shape == (2, 84, 84)  # Empilhamento (2*C, 84, 84)

    agent = UniversalS1Agent(obs_space=wrapped.observation_space, action_space=wrapped.action_space, is_visual=True)
    agent.eval()

    action = agent.act_fast(obs_dict)
    assert isinstance(action, int)
    assert action in [0, 1]

    next_obs_dict, rew, term, trunc, _ = wrapped.step(action)
    assert next_obs_dict["obs"].shape == (2, 84, 84)
    assert next_obs_dict["prev_action"] == action
    assert next_obs_dict["prev_reward"] == rew

    env.close()


def test_memory_hook_env_vector_mode():
    """Testa Nível 2 (MemoryHookEnv) em Modo Vetorial Puro."""
    backend = BufferMemoryBackend(buffer_size=1024)

    # Schema: HP (int32 no offset 0x10), Score (float32 no offset 0x14), X (float32 no offset 0x18)
    schema = [
        MemoryField(name="hp", offset=0x10, dtype="int32", is_terminal=True),
        MemoryField(name="score", offset=0x14, dtype="float32", is_reward=True),
        MemoryField(name="x", offset=0x18, dtype="float32"),
    ]

    backend.write_value(offset=0x10, value=100, dtype="int32")
    backend.write_value(offset=0x14, value=50.0, dtype="float32")
    backend.write_value(offset=0x18, value=3.14, dtype="float32")

    env = MemoryHookEnv(
        memory_schema=schema,
        actions_map={0: None, 1: "up"},
        memory_backend=backend,
        capture_screen=False,
    )

    assert env.observation_space.shape == (3,)
    obs, info = env.reset()
    assert np.allclose(obs, [100.0, 50.0, 3.14], atol=1e-3)

    # Simula incremento de pontuação na memória
    backend.write_value(offset=0x14, value=75.0, dtype="float32")
    obs, reward, term, trunc, step_info = env.step(1)
    assert reward == 25.0  # Delta score
    assert not term
    assert np.allclose(obs, [100.0, 75.0, 3.14], atol=1e-3)

    # Simula morte do jogador (HP <= 0)
    backend.write_value(offset=0x10, value=0, dtype="int32")
    obs, reward, term, trunc, step_info = env.step(0)
    assert term  # is_terminal ativado por hp <= 0

    env.close()


def test_memory_hook_env_hybrid_mode():
    """Testa Nível 2 (MemoryHookEnv) em Modo Híbrido: Visão + Recompensa/GameOver de RAM."""
    backend = BufferMemoryBackend(buffer_size=512)
    backend.write_value(offset=0x20, value=100, dtype="int32")  # HP
    backend.write_value(offset=0x24, value=0, dtype="int32")    # Score

    mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    schema = [
        MemoryField(name="hp", offset=0x20, dtype="int32", is_terminal=True),
        MemoryField(name="score", offset=0x24, dtype="int32", is_reward=True),
    ]

    env = MemoryHookEnv(
        memory_schema=schema,
        actions_map={0: None, 1: "shoot"},
        memory_backend=backend,
        capture_screen=True,
    )
    # Injata mock_capture_source no sub-adaptador de tela
    assert env.screen_adapter is not None
    env.screen_adapter.mock_capture_source = lambda: mock_frame

    assert env.is_visual
    assert env.observation_space.shape == (1, 84, 84)

    obs, info = env.reset()
    assert obs.shape == (1, 84, 84)
    assert info["ram"]["hp"] == 100.0

    # Atualiza score na RAM
    backend.write_value(offset=0x24, value=10, dtype="int32")
    obs, reward, term, trunc, step_info = env.step(1)
    assert reward == 10.0
    assert not term

    # Conecta com UniversalS1Wrapper e UniversalS1Agent
    wrapped = UniversalS1Wrapper(env, is_visual=True)
    agent = UniversalS1Agent(obs_space=wrapped.observation_space, action_space=wrapped.action_space, is_visual=True)
    agent.eval()

    obs_dict, _ = wrapped.reset()
    action = agent.act_fast(obs_dict)
    assert action in [0, 1]

    env.close()


def test_native_engine_env_lock_step_and_speed():
    """Testa Nível 3 (NativeEngineEnv): Regime Lock-Step e alta taxa de FPS headless."""
    env = NativeEngineEnv(
        engine_type="native_sim",
        is_visual=False,
        obs_dim=6,
        action_dim=3,
        is_action_discrete=True,
    )

    assert env.observation_space.shape == (6,)
    assert env.action_space.n == 3

    obs, _ = env.reset()
    assert obs.shape == (6,)

    # Mede velocidade em regime lock-step (deve atingir milhares de FPS)
    t0 = time.perf_counter()
    n_steps = 1000
    for _ in range(n_steps):
        obs, reward, term, trunc, _ = env.step(0)
    duration = time.perf_counter() - t0

    fps = n_steps / max(1e-6, duration)
    assert fps > 5000.0, f"Taxa lock-step abaixo do esperado: {fps:.0f} FPS"

    # Conecta com UniversalS1Wrapper
    wrapped = UniversalS1Wrapper(env, is_visual=False)
    agent = UniversalS1Agent(obs_space=wrapped.observation_space, action_space=wrapped.action_space, is_visual=False)
    agent.eval()

    obs_dict, _ = wrapped.reset()
    action = agent.act_fast(obs_dict)
    assert action in [0, 1, 2]

    env.close()


def test_factory_make_game_env():
    """Testa factory.py make_game_env para os 3 níveis."""
    # Nível 1: Window
    w_env = make_game_env(
        "window",
        {
            "window_bbox": {"top": 0, "left": 0, "width": 100, "height": 100},
            "actions_map": {0: None, 1: "x"},
            "mock_capture_source": lambda: np.zeros((100, 100, 3), dtype=np.uint8),
        },
    )
    assert isinstance(w_env, UniversalS1Wrapper)
    assert w_env.is_visual
    w_obs, _ = w_env.reset()
    assert w_obs["obs"].shape == (2, 84, 84)
    w_env.close()

    # Nível 2: Memory (Vetorial)
    backend = BufferMemoryBackend()
    backend.write_value(0, 1.23, "float32")
    m_env = make_game_env(
        "memory",
        {
            "memory_schema": [{"name": "val", "offset": 0, "dtype": "float32"}],
            "actions_map": {0: None},
            "memory_backend": backend,
            "capture_screen": False,
        },
    )
    assert isinstance(m_env, UniversalS1Wrapper)
    assert not m_env.is_visual
    m_obs, _ = m_env.reset()
    assert m_obs["obs"].shape == (1,)
    m_env.close()

    # Nível 3: Native (Lock-Step)
    n_env = make_game_env(
        "native",
        {
            "engine_type": "native_sim",
            "is_visual": True,
            "channels": 1,
            "action_dim": 4,
        },
    )
    assert isinstance(n_env, UniversalS1Wrapper)
    assert n_env.is_visual
    n_obs, _ = n_env.reset()
    assert n_obs["obs"].shape == (2, 84, 84)
    n_env.close()

    # Nível 3: Native ViZDoom Real
    vzd_env = make_game_env(
        "native",
        {
            "engine_type": "vizdoom",
            "scenario_path": "basic.cfg",
            "args": {"headless": True, "frame_skip": 4},
            "is_visual": True,
            "channels": 1,
        },
    )
    assert isinstance(vzd_env, UniversalS1Wrapper)
    assert vzd_env.is_visual
    v_obs, _ = vzd_env.reset()
    assert v_obs["obs"].shape == (2, 84, 84)
    frame = vzd_env.render()
    assert frame is not None
    assert frame.shape == (240, 320, 3)
    next_obs, rew, term, trunc, _ = vzd_env.step(0)
    assert next_obs["obs"].shape == (2, 84, 84)
    vzd_env.close()

    # Erro com tipo inválido
    with pytest.raises(ValueError):
        make_game_env("invalid_adapter", {})


def test_build_environment_render_mode():
    """Valida a instanciação de ambientes com flag de renderização em tempo real."""
    from system1_engine.cli import build_environment

    # Headless padrão
    env_headless = build_environment("CartPole-v1", render=False)
    assert env_headless.env.render_mode is None
    env_headless.close()

    # Com renderização ativa (render_mode='human')
    env_render = build_environment("CartPole-v1", render=True)
    assert env_render.env.render_mode == "human"
    obs, _ = env_render.reset()
    assert obs["obs"].shape == (4,)
    env_render.close()


def test_vizdoom_aim_reward_wrapper_strafe_and_rotational():
    """Valida o VizdoomAimRewardWrapper adaptando-se a cenários laterais e rotacionais 360."""
    from system1_engine.env.adapters.vizdoom import VizdoomAimRewardWrapper

    # 1. Cenário Lateral (basic.cfg)
    vzd_strafe = make_game_env(
        "native",
        {
            "engine_type": "vizdoom",
            "scenario_path": "basic.cfg",
            "args": {"headless": True, "frame_skip": 4},
        },
    )
    wrapper_strafe = VizdoomAimRewardWrapper(vzd_strafe)
    obs, info = wrapper_strafe.reset()
    assert wrapper_strafe._button_names == ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]
    next_obs, r, term, trunc, _ = wrapper_strafe.step(0)
    assert isinstance(r, float)
    assert "prev_reward" in next_obs
    assert next_obs["prev_reward"] == r
    wrapper_strafe.close()

    # 2. Cenário Rotacional 360° (defend_the_center.cfg)
    vzd_rot = make_game_env(
        "native",
        {
            "engine_type": "vizdoom",
            "scenario_path": "defend_the_center.cfg",
            "args": {"headless": True, "frame_skip": 4},
        },
    )
    wrapper_rot = VizdoomAimRewardWrapper(vzd_rot)
    obs, info = wrapper_rot.reset()
    assert wrapper_rot._button_names == ["TURN_LEFT", "TURN_RIGHT", "ATTACK"]
    # Inicialmente no defend_the_center, há um monstro alinhado na frente (delta=0)
    next_obs, r_shot, term, trunc, _ = wrapper_rot.step(2)  # ATTACK
    assert r_shot == 5.0  # Alinhado com alvo frontal
    next_obs, r_turn, term, trunc, _ = wrapper_rot.step(0)  # TURN_LEFT
    assert isinstance(r_turn, float)
    wrapper_rot.close()


