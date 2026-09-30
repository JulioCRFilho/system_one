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
