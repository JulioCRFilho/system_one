"""Testes unitários e de integração para o FrozenLake com Curriculum Learning."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.adapters.frozenlake import FrozenLakeCurriculumWrapper
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.training.ppo import RecurrentPPOTrainer


class TestFrozenLakeCurriculum:
    """Valida o funcionamento do wrapper de Curriculum Learning Reverso para o FrozenLake."""

    def test_curriculum_initial_state_and_levels(self):
        raw_env = gym.make("FrozenLake-v1", is_slippery=False)
        env = FrozenLakeCurriculumWrapper(raw_env, curriculum=True, min_level=1, max_level=4)
        assert env.current_level == 1

        obs, info = env.reset(seed=42)
        assert obs in [14, 10]  # Spawn do nível 1
        assert info["current_level"] == 1
        assert info["spawn_tile"] == obs

    def test_level_promotion_on_success(self):
        raw_env = gym.make("FrozenLake-v1", is_slippery=False)
        env = FrozenLakeCurriculumWrapper(
            raw_env,
            curriculum=True,
            min_level=1,
            max_level=3,
            target_success_rate=0.80,
            curriculum_window=5,
        )
        assert env.current_level == 1

        # Simula 5 sucessos consecutivos
        for _ in range(5):
            env.recent_successes.append(1.0)

        obs, info = env.reset(seed=123)
        assert env.current_level == 2
        assert info["current_level"] == 2
        assert obs in [13, 9, 6]  # Spawn do nível 2

    def test_reward_shaping(self):
        raw_env = gym.make("FrozenLake-v1", is_slippery=False)
        env = FrozenLakeCurriculumWrapper(raw_env, curriculum=False)
        env.reset()

        # Posiciona em 14 e move RIGHT (2) para 15 (Goal)
        env.unwrapped.s = 14
        obs, reward, terminated, truncated, info = env.step(2)
        assert obs == 15
        assert terminated is True
        assert info["is_goal"] is True
        assert reward > 1.0  # +2.0 goal_reward - 0.01 time_penalty

    def test_integration_with_universal_wrapper(self):
        raw_env = gym.make("FrozenLake-v1", is_slippery=False)
        curr_env = FrozenLakeCurriculumWrapper(raw_env, curriculum=True)
        wrapped = UniversalS1Wrapper(curr_env)

        agent = UniversalS1Agent(
            obs_space=wrapped.observation_space,
            action_space=wrapped.action_space,
        )

        obs_dict, info = wrapped.reset(seed=10)
        assert obs_dict["obs"].shape == (16,)
        assert "delta_obs" in obs_dict
        assert obs_dict["prev_action"] == 0

        # Faz inferência reflexa rápida
        dec = agent.act_fast(obs_dict, return_decision=True)
        assert dec.action in [0, 1, 2, 3]

        next_obs, r, term, trunc, step_info = wrapped.step(dec.action)
        assert next_obs["obs"].shape == (16,)
        assert isinstance(r, float)
