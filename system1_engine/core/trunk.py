from typing import Optional, Tuple
import torch
import torch.nn as nn


class ResMLPBlock(nn.Module):
    """Residual Multi-Layer Perceptron Block.

    Structure:
      x + Linear(GELU(Linear(LayerNorm(x))))
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return res + x


class System1Trunk(nn.Module):
    """System 1 Recurrent Trunk.

    Universal & transferable reflexive core with latent recurrence:
      - nn.LayerNorm(337)
      - nn.GRU(337 -> 256, batch_first=True)
      - 2x ResMLP Blocks (256)

    Strictly decoupled from front-ends and decision heads.
    Can be frozen and transferred across environments.
    """

    def __init__(
        self,
        input_dim: int = 337,
        hidden_dim: int = 256,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.res_blocks = nn.Sequential(
            *[ResMLPBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through recurrent trunk.

        Args:
            x: Input tensor of shape [B, T, 337] or [B, 337].
            hx: Hidden state of GRU [1, B, 256].
            dones: Episode boundary mask [B, T]. If provided, zeroes hx at episode resets.

        Returns:
            Tuple of:
              - h: Latent representations [B, T, 256]
              - next_hx: Updated GRU hidden state [1, B, 256]
        """
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)  # [B, 1, 337]

        b, t, d = x.shape
        assert d == self.input_dim, (
            f"Expected input dimension {self.input_dim}, got {d}"
        )

        if hx is not None:
            assert hx.shape == (1, b, self.hidden_dim), (
                f"Expected hx shape {(1, b, self.hidden_dim)}, got {hx.shape}"
            )
        else:
            hx = torch.zeros(1, b, self.hidden_dim, device=x.device, dtype=x.dtype)

        # Normalize input
        x_norm = self.input_norm(x)

        if dones is None:
            # Fast vectorized unroll
            gru_out, next_hx = self.gru(x_norm, hx)
        else:
            # Temporal masking unroll to prevent cross-trajectory state leakage
            assert dones.shape == (b, t), (
                f"Expected dones shape {(b, t)}, got {dones.shape}"
            )
            step_outs = []
            curr_h = hx
            for step in range(t):
                # Mask hidden state immediately before step t
                mask = (1.0 - dones[:, step].float()).view(1, b, 1)
                curr_h = curr_h * mask
                step_x = x_norm[:, step : step + 1, :]
                step_out, curr_h = self.gru(step_x, curr_h)
                step_outs.append(step_out)

            gru_out = torch.cat(step_outs, dim=1)
            next_hx = curr_h

        # ResMLP processing
        h = self.res_blocks(gru_out)
        assert h.shape == (b, t, self.hidden_dim), (
            f"Expected h shape {(b, t, self.hidden_dim)}, got {h.shape}"
        )

        if is_2d:
            h = h.squeeze(1)

        return h, next_hx
