import os
import gymnasium as gym
import numpy as np
import pytest
import torch

from system1_engine.cli import build_environment
from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.adapters.mountain_car import (
    MountainCarEnergyRewardWrapper,
    MountainCarNormalizedWrapper,
)
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.hud.worker import build_hud_env


def test_mountain_car_normalized_wrapper():
    raw_env = gym.make("MountainCar-v0")
    norm_env = MountainCarNormalizedWrapper(raw_env)

    # Observation space bounds should be updated for velocity
    assert norm_env.observation_space.low[1] == -1.0
    assert norm_env.observation_space.high[1] == 1.0

    obs, _ = norm_env.reset(seed=42)
    assert len(obs) == 2
    # Velocity scaled by 14.2857
    raw_vel = norm_env.unwrapped.state[1]
    assert np.isclose(obs[1], raw_vel * 14.2857, atol=1e-4)

    obs, reward, term, trunc, _ = norm_env.step(2)
    raw_vel = norm_env.unwrapped.state[1]
    assert np.isclose(obs[1], raw_vel * 14.2857, atol=1e-4)
    # Standard reward is untouched
    assert reward == -1.0
    norm_env.close()


def test_mountain_car_energy_reward_wrapper():
    raw_env = gym.make("MountainCar-v0")
    energy_env = MountainCarEnergyRewardWrapper(
        raw_env, power_scale=20.0, height_scale=5.0, goal_bonus=50.0
    )

    energy_env.reset(seed=42)
    # Force state: car at bottom with positive velocity
    energy_env.unwrapped.state = np.array([-0.5, 0.02], dtype=np.float64)

    # Action 2: push right (force = +1). Force * vel = +1 * 0.02 > 0 -> power bonus
    _, reward_accelerate, _, _, _ = energy_env.step(2)

    # Reset to same state
    energy_env.unwrapped.state = np.array([-0.5, 0.02], dtype=np.float64)
    # Action 0: push left (force = -1). Force * vel = -1 * 0.02 < 0 -> braking penalty
    _, reward_brake, _, _, _ = energy_env.step(0)

    # Pushing in the direction of velocity should give strictly higher reward
    assert reward_accelerate > reward_brake

    # Test goal bonus: force reaching goal
    energy_env.unwrapped.state = np.array([0.5, 0.01], dtype=np.float64)
    _, reward_goal, term, _, _ = energy_env.step(2)
    assert term is True
    # Reward should include the goal bonus (+50)
    assert reward_goal > 40.0
    energy_env.close()


def test_cli_build_environment_mountain_car():
    # Training mode should have shaping + normalization
    train_env = build_environment("MountainCar-v0", is_training=True)
    assert isinstance(train_env, UniversalS1Wrapper)
    # Inner environment should have MountainCarNormalizedWrapper
    assert isinstance(train_env.env, MountainCarNormalizedWrapper)
    # Next inner should have MountainCarEnergyRewardWrapper
    assert isinstance(train_env.env.env, MountainCarEnergyRewardWrapper)
    train_env.close()

    # Eval / Run mode should have normalization ONLY (unshaped reward)
    eval_env = build_environment("MountainCar-v0", is_training=False)
    assert isinstance(eval_env, UniversalS1Wrapper)
    assert isinstance(eval_env.env, MountainCarNormalizedWrapper)
    # Reward wrapper should NOT be present in eval mode
    assert not isinstance(eval_env.env.env, MountainCarEnergyRewardWrapper)
    eval_env.close()


def test_hud_build_hud_env_mountain_car():
    train_env = build_hud_env("MountainCar-v0", render_mode="in_browser", is_training=True)
    assert isinstance(train_env.env, MountainCarNormalizedWrapper)
    assert isinstance(train_env.env.env, MountainCarEnergyRewardWrapper)
    train_env.close()

    eval_env = build_hud_env("MountainCar-v0", render_mode="in_browser", is_training=False)
    assert isinstance(eval_env.env, MountainCarNormalizedWrapper)
    assert not isinstance(eval_env.env.env, MountainCarEnergyRewardWrapper)
    eval_env.close()


def test_cartpole_unaffected():
    # CartPole must not be wrapped by MountainCar wrappers
    cp_train = build_environment("CartPole-v1", is_training=True)
    assert not isinstance(cp_train.env, MountainCarNormalizedWrapper)
    cp_train.close()

    cp_eval = build_environment("CartPole-v1", is_training=False)
    assert not isinstance(cp_eval.env, MountainCarNormalizedWrapper)
    cp_eval.close()


@pytest.mark.skipif(
    not os.path.exists("s1_mountaincar_v0_trained.pt"),
    reason="Checkpoint pré-treinado não encontrado (limpo para retreino)",
)
def test_trained_checkpoint_reaches_goal():
    eval_env = build_environment("MountainCar-v0", is_training=False)
    agent = UniversalS1Agent(
        obs_space=eval_env.env.observation_space,
        action_space=eval_env.action_space,
    )
    checkpoint = torch.load("s1_mountaincar_v0_trained.pt", map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    agent.load_state_dict(state_dict, strict=False)
    agent.eval()

    obs, _ = eval_env.reset(seed=123)
    agent.reset_memory()
    total_reward = 0.0
    done = False
    steps = 0
    while not done:
        action = agent.act_fast(obs)
        obs, r, term, trunc, _ = eval_env.step(action)
        total_reward += r
        steps += 1
        done = term or trunc

    # Must solve in < 200 steps (not timeout) and reach goal
    assert steps < 200
    assert total_reward > -200.0
    eval_env.close()
