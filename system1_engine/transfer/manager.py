from typing import Any, Dict, List, Optional
import os
import torch
import torch.nn as nn


class KnowledgeTransferManager:
    """Manages cross-scenario knowledge transfer and catastrophic forgetting prevention.

    Decouples perception and action semantics by isolating, saving, and selectively
    loading the universal System 1 reflexive trunk.
    """

    @staticmethod
    def save_checkpoint(
        agent: nn.Module,
        checkpoint_path: str,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves agent weights and metadata to file."""
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        payload = {
            "state_dict": agent.state_dict(),
            "extra_info": extra_info or {},
        }
        torch.save(payload, checkpoint_path)

    @staticmethod
    def load_transferable_weights(
        agent: Any,
        checkpoint_path: str,
        freeze_trunk: bool = True,
    ) -> List[str]:
        """Loads exclusively the trunk parameters from a checkpoint and optionally freezes them.

        Args:
            agent: The UniversalS1Agent instance.
            checkpoint_path: Path to .pt checkpoint file.
            freeze_trunk: If True, sets requires_grad = False for all trunk parameters.

        Returns:
            List of loaded trunk parameter keys.
        """
        assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            raw_state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            raw_state_dict = checkpoint
        else:
            raise ValueError(f"Unrecognized checkpoint format at {checkpoint_path}")

        # Filter strictly for trunk parameters
        trunk_dict: Dict[str, torch.Tensor] = {}
        for key, value in raw_state_dict.items():
            if key.startswith("trunk."):
                sub_key = key[len("trunk.") :]
                trunk_dict[sub_key] = value
            elif not any(key.startswith(p) for p in ["front_end.", "policy_head.", "value_head."]):
                # If checkpoint saved trunk alone
                trunk_dict[key] = value

        assert len(trunk_dict) > 0, "No trunk parameters found in checkpoint"

        # Load weights into trunk
        agent.trunk.load_state_dict(trunk_dict, strict=True)
        loaded_keys = list(trunk_dict.keys())

        # Freeze trunk if requested
        if freeze_trunk:
            for param in agent.trunk.parameters():
                param.requires_grad = False

            # Defensive verification that all trunk params are frozen
            for name, param in agent.trunk.named_parameters():
                assert not param.requires_grad, f"Trunk parameter {name} was not frozen!"

        return loaded_keys

    @staticmethod
    def inspect_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        """Inspects a checkpoint and extracts metadata including environment, observation and action dimensions."""
        if not os.path.exists(checkpoint_path):
            return {"error": f"Arquivo não encontrado: {checkpoint_path}", "path": checkpoint_path}

        info: Dict[str, Any] = {
            "path": checkpoint_path,
            "filename": os.path.basename(checkpoint_path),
            "env_id": None,
            "obs_dim": None,
            "act_dim": None,
            "steps": None,
            "final_return": None,
            "is_visual": False,
        }
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            extra = checkpoint.get("extra_info", {}) if isinstance(checkpoint, dict) else {}
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else {}

            info["env_id"] = extra.get("env_id")
            info["final_return"] = extra.get("final_return")
            info["steps"] = extra.get("steps")

            # Extract observation dimension
            if "front_end.state_encoder.net.0.weight" in state_dict:
                info["obs_dim"] = int(state_dict["front_end.state_encoder.net.0.weight"].shape[1])
                info["is_visual"] = False
            elif any("visual_encoder" in k for k in state_dict.keys()):
                info["obs_dim"] = "visual"
                info["is_visual"] = True

            # Extract action dimension
            if "policy_head.linear.weight" in state_dict:
                info["act_dim"] = int(state_dict["policy_head.linear.weight"].shape[0])

            # Deduce env_id from filename if not in metadata
            if not info["env_id"]:
                fn = os.path.basename(checkpoint_path).lower()
                if "cartpole" in fn:
                    info["env_id"] = "CartPole-v1"
                elif "acrobot" in fn:
                    info["env_id"] = "Acrobot-v1"
                elif "pendulum" in fn:
                    info["env_id"] = "Pendulum-v1"
                elif "mountaincar" in fn:
                    info["env_id"] = "MountainCar-v0"
                elif "vizdoom" in fn or "doom" in fn:
                    info["env_id"] = "vizdoom"

        except Exception as e:
            info["error"] = str(e)

        return info

    @staticmethod
    def validate_evaluation_compatibility(
        agent: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        env_id: str,
        checkpoint_path: str,
    ) -> Optional[str]:
        """Verifies if a state_dict can be loaded for direct evaluation without shape mismatches.

        Returns None if compatible, or an error string describing the mismatch.
        """
        mismatches: List[str] = []
        agent_dict = agent.state_dict()
        for k, v in state_dict.items():
            if k in agent_dict and isinstance(v, torch.Tensor):
                if agent_dict[k].shape != v.shape:
                    mismatches.append(f"  • {k}: salvo={list(v.shape)}, esperado={list(agent_dict[k].shape)}")

        if mismatches:
            ckpt_info = KnowledgeTransferManager.inspect_checkpoint(checkpoint_path)
            orig_env = ckpt_info.get("env_id") or "outro ambiente"
            msg = (
                f"\n{'='*72}\n"
                f"❌ [ERRO DE COMPATIBILIDADE DE CHECKPOINT]\n"
                f"O checkpoint '{checkpoint_path}' é incompatível com o ambiente '{env_id}' para AVALIAÇÃO direta!\n\n"
                f"🔍 Tensores com divergência de formato (Size Mismatch):\n"
                + "\n".join(mismatches[:5])
                + (f"\n  ... (+ {len(mismatches) - 5} tensores)" if len(mismatches) > 5 else "")
                + f"\n\n💡 DIAGNÓSTICO E COMO RESOLVER:\n"
                f"  1. AVALIAÇÃO DIRETA: O modo 'Avaliar' requer pesos treinados no mesmo ambiente.\n"
                f"     -> No menu 'Ambiente (--env)', selecione '{orig_env}'.\n"
                f"  2. TRANSFERÊNCIA DE APRENDIZADO (Cross-Task):\n"
                f"     -> Para aproveitar os 18 tensores do tronco de '{checkpoint_path}' em '{env_id}',\n"
                f"        use o modo 'Treino' com 'Transferir Tronco (--transfer-from)' e 'Congelar Tronco'.\n"
                f"{'='*72}\n"
            )
            return msg
        return None
