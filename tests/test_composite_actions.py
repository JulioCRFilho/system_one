import gymnasium as gym
import numpy as np
import pytest

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.adapters import make_game_env
from system1_engine.env.adapters.native_adapter import NativeEngineEnv
from system1_engine.training.ppo import RecurrentPPOTrainer


def test_build_composite_actions_logic():
    """Valida a geração ortogonal de ações combinadas para diferentes configurações de botões."""
    # 1. Deadly corridor (7 botões -> 54 ações)
    corridor_buttons = [
        "MOVE_LEFT", "MOVE_RIGHT", "ATTACK", "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"
    ]
    table_54, descs_54 = NativeEngineEnv._build_composite_actions(corridor_buttons)
    assert len(table_54) == 54
    assert len(descs_54) == 54

    # Verifica combinações fundamentais exigidas
    assert "NO_OP" in descs_54
    assert "ATTACK" in descs_54
    assert "MOVE_FORWARD" in descs_54
    assert "MOVE_FORWARD+ATTACK" in descs_54
    assert "MOVE_RIGHT+TURN_LEFT+ATTACK" in descs_54
    assert "MOVE_FORWARD+MOVE_RIGHT+TURN_LEFT+ATTACK" in descs_54
    assert "MOVE_BACKWARD+MOVE_RIGHT+TURN_LEFT+ATTACK" in descs_54

    # Valida vetor correspondente à ação solicitada pelo usuário (TURN_LEFT + MOVE_RIGHT + ATTACK)
    idx = descs_54.index("MOVE_RIGHT+TURN_LEFT+ATTACK")
    vec = table_54[idx]
    # Espera 1 nos índices de MOVE_RIGHT, ATTACK, TURN_LEFT e 0 nos demais
    for btn_name, pressed in zip(corridor_buttons, vec):
        if btn_name in ["MOVE_RIGHT", "ATTACK", "TURN_LEFT"]:
            assert pressed == 1
        else:
            assert pressed == 0

    # 2. Cenário Lateral Strafe (basic.cfg: 3 botões -> 6 ações)
    basic_buttons = ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]
    table_6, descs_6 = NativeEngineEnv._build_composite_actions(basic_buttons)
    assert len(table_6) == 6
    assert "MOVE_LEFT+ATTACK" in descs_6
    assert "MOVE_RIGHT+ATTACK" in descs_6

    # 3. Cenário Rotacional (defend_the_center.cfg: 3 botões -> 6 ações)
    rot_buttons = ["TURN_LEFT", "TURN_RIGHT", "ATTACK"]
    table_rot, descs_rot = NativeEngineEnv._build_composite_actions(rot_buttons)
    assert len(table_rot) == 6
    assert "TURN_LEFT+ATTACK" in descs_rot
    assert "TURN_RIGHT+ATTACK" in descs_rot


def test_native_engine_vizdoom_composite_execution():
    """Valida execução em tempo real de ações compostas e sinalizador de ataque no ViZDoom."""
    env = NativeEngineEnv(
        engine_type="vizdoom",
        scenario_path="deadly_corridor.cfg",
        args={"headless": True, "frame_skip": 4, "combined_actions": True, "reward_shaping": True},
        is_visual=True,
        channels=1,
    )

    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 54

    # Identifica ação composta com disparo: MOVE_RIGHT + TURN_LEFT + ATTACK
    descs = env.action_descriptions
    act_name = "MOVE_RIGHT+TURN_LEFT+ATTACK"
    assert act_name in descs
    act_idx = descs.index(act_name)

    obs, _ = env.reset()
    assert obs.shape == (1, 84, 84)

    next_obs, reward, term, trunc, info = env.step(act_idx)
    assert env.is_attacking is True
    assert info["action_name"] == act_name
    assert info["action_vector"] == env.action_table[act_idx]
    assert isinstance(reward, float)

    env.close()


def test_end_to_end_agent_ppo_with_composite_actions():
    """Valida o pipeline completo de aprendizado PPO com 54 ações discretas combinadas."""
    env = make_game_env(
        "native",
        {
            "engine_type": "vizdoom",
            "scenario_path": "deadly_corridor.cfg",
            "args": {"headless": True, "frame_skip": 4, "combined_actions": True, "reward_shaping": True},
            "is_visual": True,
            "channels": 1,
        },
    )

    assert env.action_space.n == 54

    agent = UniversalS1Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        is_visual=True,
    )
    assert agent.policy_head.n_actions == 54

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        rollout_steps=16,
        chunk_length=8,
        chunk_batch_size=2,
        device="cpu",
    )

    obs_dict, _ = env.reset()
    obs_dict, hx, ep_start, ret = trainer.collect_rollouts(obs_dict, None, True)
    assert trainer.total_steps == 16

    metrics = trainer.train_epoch()
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy_loss" in metrics

    env.close()
