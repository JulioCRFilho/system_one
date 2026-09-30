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


def test_checkpoint_inspection_and_evaluation_compatibility(tmp_path):
    """Ensure inspect_checkpoint reads metadata and validate_evaluation_compatibility detects dimension mismatches."""
    # 1. Save an Acrobot checkpoint
    acro_env = UniversalS1Wrapper(gym.make("Acrobot-v1"))
    acro_agent = UniversalS1Agent(acro_env.env.observation_space, acro_env.action_space)
    acro_path = str(tmp_path / "s1_acrobot.pt")
    KnowledgeTransferManager.save_checkpoint(
        acro_agent,
        acro_path,
        extra_info={"env_id": "Acrobot-v1", "final_return": -85.0, "steps": 5000},
    )

    # 2. Inspect checkpoint
    meta = KnowledgeTransferManager.inspect_checkpoint(acro_path)
    assert meta["env_id"] == "Acrobot-v1"
    assert meta["obs_dim"] == 6
    assert meta["act_dim"] == 3
    assert meta["final_return"] == -85.0
    assert meta["steps"] == 5000

    # 3. Create a CartPole agent (obs=4, act=2)
    cart_env = UniversalS1Wrapper(gym.make("CartPole-v1"))
    cart_agent = UniversalS1Agent(cart_env.env.observation_space, cart_env.action_space)

    # 4. Check compatibility: evaluating an Acrobot checkpoint on CartPole must return an informative error message
    checkpoint = torch.load(acro_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    err = KnowledgeTransferManager.validate_evaluation_compatibility(cart_agent, state_dict, "CartPole-v1", acro_path)
    assert err is not None
    assert "ERRO DE COMPATIBILIDADE DE CHECKPOINT" in err
    assert "front_end.state_encoder.net.0.weight" in err
    assert "Acrobot-v1" in err

    # 5. But evaluating it on an Acrobot agent must return None (compatible!)
    err_compat = KnowledgeTransferManager.validate_evaluation_compatibility(acro_agent, state_dict, "Acrobot-v1", acro_path)
    assert err_compat is None
