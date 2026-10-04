#!/usr/bin/env python3
"""Exportador de Checkpoints PyTorch (.pt) para ONNX (.onnx) no System 1 Engine.

Permite que os pesos do agente sejam consumidos diretamente no navegador
pelo repositório do portfólio via ONNX Runtime Web (Wasm/WebGPU).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import torch
import torch.nn as nn

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.adapters.rubiks import RubiksCubeMacroEnv, RubiksCubeEnv
from system1_engine.env.wrapper import UniversalS1Wrapper


class S1OnnxExportWrapper(nn.Module):
    """Wrapper para exportação simbólica ONNX pura sem classes dinâmicas de distribuição."""

    def __init__(self, agent: UniversalS1Agent) -> None:
        super().__init__()
        self.front_end = agent.front_end
        self.trunk = agent.trunk
        self.policy_head = agent.policy_head
        self.value_head = agent.value_head
        self.is_discrete = agent.is_discrete

    def forward(
        self,
        obs: torch.Tensor,
        delta_obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        hx: torch.Tensor,
    ):
        z_in = self.front_end(
            obs=obs,
            delta_obs=delta_obs,
            prev_action=prev_action,
            prev_reward=prev_reward,
        )
        h, next_hx = self.trunk(z_in, hx=hx)
        if self.is_discrete:
            logits = self.policy_head.linear(h).squeeze(1)
        else:
            logits = self.policy_head.mu_net(h).squeeze(1)
        val = self.value_head(h).squeeze(1)
        return logits, val, next_hx


def export_checkpoint(checkpoint_path: str, env_id: str, output_path: str) -> None:
    print(f"📦 Exportando '{checkpoint_path}' para '{output_path}'...")
    
    # 1. Cria o ambiente para resolver espaços de observação e ação
    env_lower = env_id.lower().replace("_", "-")
    if env_lower in ("rubiks", "rubik", "rubiksmacro", "rubiks-macro", "rubikscubemacro-v0"):
        raw_env = RubiksCubeMacroEnv()
    elif env_lower in ("rubiks-atomic", "rubik-atomic", "rubikscube-v0"):
        raw_env = RubiksCubeEnv()
    else:
        raw_env = gym.make(env_id)

    wrapped_env = UniversalS1Wrapper(raw_env)
    obs_space = wrapped_env.observation_space
    action_space = wrapped_env.action_space
    obs_dim = obs_space["obs"].shape[0]

    # 2. Inicializa o agente e carrega os pesos
    agent = UniversalS1Agent(obs_space=obs_space, action_space=action_space)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        agent.load_state_dict(state_dict, strict=False)
        print(f"  • Pesos carregados com sucesso de: {checkpoint_path}")
    else:
        print(f"  ⚠️ Checkpoint '{checkpoint_path}' não encontrado. Exportando com pesos aleatórios.")

    agent.eval()
    export_model = S1OnnxExportWrapper(agent)
    export_model.eval()

    # 3. Tensors dummy de exemplo (batch=1, seq=1)
    dummy_obs = torch.zeros(1, 1, obs_dim, dtype=torch.float32)
    dummy_delta = torch.zeros(1, 1, obs_dim, dtype=torch.float32)
    dummy_act = torch.zeros(1, 1, dtype=torch.long if agent.is_discrete else torch.float32)
    dummy_rew = torch.zeros(1, 1, 1, dtype=torch.float32)
    dummy_hx = torch.zeros(1, 1, 256, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Usa exportador clássico (dynamo=False) para embutir todos os pesos
    # em um único arquivo binário auto-contido, facilitando importação web/CDN direta
    torch.onnx.export(
        export_model,
        (dummy_obs, dummy_delta, dummy_act, dummy_rew, dummy_hx),
        output_path,
        input_names=["obs", "delta_obs", "prev_action", "prev_reward", "hx"],
        output_names=["logits", "value", "next_hx"],
        opset_version=17,
        dynamo=False,
    )
    print(f"✅ Sucesso! Modelo ONNX gerado: {output_path} ({os.path.getsize(output_path):,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Exportador PyTorch -> ONNX do System 1")
    parser.add_argument("--checkpoint", type=str, required=True, help="Caminho do arquivo .pt")
    parser.add_argument("--env", type=str, required=True, help="ID do ambiente (ex: RubiksCubeMacro-v0, CartPole-v1)")
    parser.add_argument("--out", type=str, required=True, help="Caminho de saída .onnx")
    args = parser.parse_args()

    export_checkpoint(args.checkpoint, args.env, args.out)


if __name__ == "__main__":
    main()
