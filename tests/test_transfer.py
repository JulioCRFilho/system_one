import os
import gymnasium as gym
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.transfer.manager import KnowledgeTransferManager


def test_transfer_without_catastrophic_forgetting(tmp_path):
    """Verify cross-scenario transfer: weights loaded from CartPole to Acrobot remain frozen and identical."""
    # 1. Train or save a checkpoint on CartPole
    ckpt_path = str(tmp_path / "s1_cartpole.pt")
    if os.path.exists("s1_cartpole.pt"):
        # Use existing verified checkpoint
        ckpt_path = "s1_cartpole.pt"
    else:
        # Create a fresh agent and save state
        cart_env = UniversalS1Wrapper(gym.make("CartPole-v1"))
        cart_agent = UniversalS1Agent(cart_env.env.observation_space, cart_env.action_space)
        KnowledgeTransferManager.save_checkpoint(cart_agent, ckpt_path)

    # 2. Initialize agent in Acrobot-v1 (obs_dim=6, act_dim=3)
    acrobot_raw = gym.make("Acrobot-v1")
    acrobot_env = UniversalS1Wrapper(acrobot_raw)

    acrobot_agent = UniversalS1Agent(
        obs_space=acrobot_env.env.observation_space,
        action_space=acrobot_env.action_space,
    )

    # Confirm observation shapes differ
    assert acrobot_env.env.observation_space.shape[0] == 6

    # 3. Load transferable trunk weights with freeze_trunk=True
    loaded_keys = KnowledgeTransferManager.load_transferable_weights(
        agent=acrobot_agent,
        checkpoint_path=ckpt_path,
        freeze_trunk=True,
    )
    assert len(loaded_keys) > 0, "No trunk weights loaded"

    # 4. Verify all trunk parameters have requires_grad == False and record initial values
    initial_trunk_weights = {}
    for name, param in acrobot_agent.trunk.named_parameters():
        assert not param.requires_grad, f"Trunk param {name} requires_grad must be False"
        initial_trunk_weights[name] = param.detach().clone()

    # Verify non-trunk parameters DO require grad
    for name, param in acrobot_agent.front_end.named_parameters():
        assert param.requires_grad, f"Front-end param {name} should be trainable"
    for name, param in acrobot_agent.policy_head.named_parameters():
        assert param.requires_grad, f"Policy head param {name} should be trainable"

    # 5. Train 5 epochs on Acrobot-v1
    trainer = RecurrentPPOTrainer(
        agent=acrobot_agent,
        env=acrobot_env,
        learning_rate=1e-3,
        rollout_steps=1024,
        chunk_length=16,
        chunk_batch_size=16,
        n_epochs=5,
    )

    obs_dict, _ = acrobot_env.reset()
    obs_dict, hx, ep_start, _ = trainer.collect_rollouts(obs_dict, None, True)
    trainer.train_epoch()

    # 6. CRITICAL ACCEPTANCE CHECK:
    # Verify all trunk parameters remain strictly requires_grad == False and bitwise identical
    for name, param in acrobot_agent.trunk.named_parameters():
        assert not param.requires_grad, f"Trunk param {name} requires_grad became True after training"
        diff = torch.max(torch.abs(param - initial_trunk_weights[name])).item()
        assert diff == 0.0, f"Trunk param {name} changed numerically by {diff}!"
