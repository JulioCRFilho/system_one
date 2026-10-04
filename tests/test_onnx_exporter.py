"""Testes unitários para exportação automática ONNX e sincronização Web/Portfólio."""

from __future__ import annotations

import os
import tempfile
import gymnasium as gym
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.core.onnx_exporter import auto_sync_web_model, resolve_web_model_filename
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.transfer.manager import KnowledgeTransferManager


def test_resolve_web_model_filename():
    assert resolve_web_model_filename("CartPole-v1") == "s1_cartpole.onnx"
    assert resolve_web_model_filename("RubiksCubeMacro-v0") == "s1_rubiks_macro.onnx"
    assert resolve_web_model_filename("RubiksCube-v0") == "s1_rubiks_atomic.onnx"
    assert resolve_web_model_filename(checkpoint_path="s1_rubiks_macro_trained.pt") == "s1_rubiks_macro.onnx"
    assert resolve_web_model_filename(checkpoint_path="s1_cartpole.pt") == "s1_cartpole.onnx"


def test_auto_sync_web_model_vector():
    env = gym.make("CartPole-v1")
    wrapped = UniversalS1Wrapper(env)
    agent = UniversalS1Agent(obs_space=wrapped.observation_space, action_space=wrapped.action_space)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "s1_testvector_v0_test.pt")
        # Deve salvar o checkpoint e disparar a exportação web
        KnowledgeTransferManager.save_checkpoint(
            agent=agent,
            checkpoint_path=ckpt_path,
            extra_info={"env_id": "TestVector-v0"},
            auto_sync_web=True,
        )
        assert os.path.exists(ckpt_path)

        # Verifica se o arquivo ONNX foi gerado
        result = auto_sync_web_model(agent, checkpoint_path=ckpt_path, extra_info={"env_id": "TestVector-v0"})
        assert result is not None
        assert os.path.exists(result)
        assert os.path.getsize(result) > 100_000

        # Limpeza
        try:
            if os.path.exists(result):
                os.remove(result)
        except OSError:
            pass
