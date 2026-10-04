"""Módulo de exportação ONNX e sincronização automática com o Web HUD e Portfólio.

Garante que sempre que um checkpoint for persistido (via HUD local, CLI ou scripts),
os pesos neurais sejam automaticamente convertidos para ONNX e sincronizados:
  1. No runtime web local do system_one (web/models/)
  2. No portfólio web oficial (../portfolio/public/models/ e ../portfolio/dist/models/)
"""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from system1_engine.core.agent import UniversalS1Agent


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


def resolve_web_model_filename(env_id: Optional[str] = None, checkpoint_path: Optional[str] = None) -> str:
    """Mapeia o ambiente ou caminho de checkpoint para o nome canônico usado no HUD Web."""
    combined = f"{env_id or ''} {checkpoint_path or ''}".lower()
    if "rubiks_atomic" in combined or "rubikscube-v0" in combined:
        return "s1_rubiks_atomic.onnx"
    if "rubik" in combined:
        return "s1_rubiks_macro.onnx"
    if "cartpole" in combined:
        return "s1_cartpole.onnx"
    if env_id:
        clean = "".join(c if c.isalnum() else "_" for c in env_id.lower()).strip("_")
        return f"s1_{clean}.onnx"
    if checkpoint_path:
        base = os.path.basename(checkpoint_path)
        name_only = os.path.splitext(base)[0]
        return f"{name_only}.onnx"
    return "s1_model.onnx"


def auto_sync_web_model(
    agent: UniversalS1Agent,
    checkpoint_path: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Exporta o agente em memória para ONNX e copia para os diretórios web e portfólio.

    Returns:
        Caminho do arquivo exportado ou None se o ambiente for incompatível.
    """
    if getattr(agent, "is_visual", False):
        # Modelos convolucionais pesados (ex: ViZDoom/CarRacing) não são exportados automaticamente para web
        return None

    try:
        env_id = (extra_info or {}).get("env_id", "")
        model_filename = resolve_web_model_filename(env_id, checkpoint_path)

        # 1. Encontra os diretórios de destino
        project_root = Path(__file__).resolve().parent.parent.parent
        web_models_dir = project_root / "web" / "models"
        web_models_dir.mkdir(parents=True, exist_ok=True)
        primary_onnx_path = web_models_dir / model_filename

        # 2. Prepara o wrapper para exportação
        agent_cpu = agent.cpu()
        agent_cpu.eval()
        export_model = S1OnnxExportWrapper(agent_cpu)
        export_model.eval()

        obs_space = getattr(agent, "obs_space", None)
        if obs_space is not None and hasattr(obs_space, "shape"):
            obs_dim = int(np.prod(obs_space.shape))
        else:
            obs_dim = getattr(agent.front_end, "obs_dim", 4)

        dummy_obs = torch.zeros(1, 1, obs_dim, dtype=torch.float32)
        dummy_delta = torch.zeros(1, 1, obs_dim, dtype=torch.float32)
        dummy_act = torch.zeros(1, 1, dtype=torch.long if agent.is_discrete else torch.float32)
        dummy_rew = torch.zeros(1, 1, 1, dtype=torch.float32)
        dummy_hx = torch.zeros(1, 1, 256, dtype=torch.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                export_model,
                (dummy_obs, dummy_delta, dummy_act, dummy_rew, dummy_hx),
                str(primary_onnx_path),
                input_names=["obs", "delta_obs", "prev_action", "prev_reward", "hx"],
                output_names=["logits", "value", "next_hx"],
                opset_version=17,
                dynamo=False,
            )

        synced_targets: List[str] = [str(primary_onnx_path)]

        # 3. Sincronização automática com o portfólio (se existir como projeto irmão)
        portfolio_candidates = [
            project_root.parent / "portfolio" / "public" / "models",
            project_root.parent / "portfolio" / "dist" / "models",
        ]

        for p_dir in portfolio_candidates:
            if p_dir.parent.exists():
                p_dir.mkdir(parents=True, exist_ok=True)
                dest = p_dir / model_filename
                shutil.copy2(primary_onnx_path, dest)
                synced_targets.append(str(dest))

        file_size_mb = primary_onnx_path.stat().st_size / (1024 * 1024)
        print(f"\n🌐 [Auto-Sync Web] Pesos ONNX sincronizados com sucesso ({file_size_mb:.2f} MB):")
        for t in synced_targets:
            print(f"   ✓ {t}")

        return str(primary_onnx_path)
    except Exception as e:
        print(f"⚠️ [Auto-Sync Web] Falha na auto-exportação ONNX (não-bloqueante): {e}")
        return None
