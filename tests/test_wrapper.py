import gymnasium as gym
import numpy as np
import pytest

from system1_engine.env.wrapper import UniversalS1Wrapper


def test_wrapper_reset_robustness():
    """Verify that on reset() after termination or truncation, all state is flushed."""
    env = UniversalS1Wrapper(gym.make("CartPole-v1"))

    # Initial reset
    obs_dict, info = env.reset()
    assert np.all(obs_dict["delta_obs"] == 0.0)
    assert obs_dict["prev_action"] == 0
    assert obs_dict["prev_reward"] == 0.0
    assert env.prev_obs is not None

    # Step until termination
    done = False
    step_count = 0
    while not done:
        action = env.action_space.sample()
        obs_dict, reward, terminated, truncated, _ = env.step(action)
        step_count += 1
        done = terminated or truncated

    # At termination, buffers contain non-zero episode data
    assert env.prev_action is not None
    assert env.prev_obs is not None

    # Reset after termination
    obs_dict_reset, info = env.reset()

    # CRITICAL: Verify zero cross-episode residual data leakage
    assert np.all(obs_dict_reset["delta_obs"] == 0.0), "delta_obs must be strictly zero at episode start"
    assert obs_dict_reset["prev_action"] == 0, "prev_action must be reset to 0"
    assert obs_dict_reset["prev_reward"] == 0.0, "prev_reward must be reset to 0.0"
    assert env.prev_reward == 0.0
    assert env.prev_action == 0
    assert np.all(env.prev_obs == obs_dict_reset["obs"])


def test_wrapper_delta_computation():
    """Verify delta_obs is mathematically s_t - s_{t-1}."""
    env = UniversalS1Wrapper(gym.make("CartPole-v1"))
    obs0_dict, _ = env.reset()
    s0 = obs0_dict["obs"]

    # Step 1
    obs1_dict, r0, _, _, _ = env.step(1)
    s1 = obs1_dict["obs"]
    expected_delta = s1 - s0

    np.testing.assert_allclose(obs1_dict["delta_obs"], expected_delta, rtol=1e-5, atol=1e-5)
    assert obs1_dict["prev_action"] == 1
    assert obs1_dict["prev_reward"] == float(r0)


def test_wrapper_continuous_action():
    """Verify wrapper works seamlessly for continuous action spaces (Box)."""
    # Create Pendulum-v1 (continuous Box action space)
    env = UniversalS1Wrapper(gym.make("Pendulum-v1"))
    obs_dict, _ = env.reset()

    assert isinstance(obs_dict["prev_action"], np.ndarray)
    assert obs_dict["prev_action"].shape == env.action_space.shape
    assert np.all(obs_dict["prev_action"] == 0.0)
    assert obs_dict["prev_reward"] == 0.0

    action = np.array([1.5], dtype=np.float32)
    next_dict, reward, _, _, _ = env.step(action)

    np.testing.assert_allclose(next_dict["prev_action"], action)
    assert next_dict["prev_reward"] == float(reward)
