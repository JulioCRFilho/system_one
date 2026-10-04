from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn

from system1_engine.core.agent import UniversalS1Agent


def _build_colormap_lut(name: str = "turbo") -> np.ndarray:
    """Gera tabela de consulta (LUT 256x3 uint8) para mapeamento rápido de calor térmico."""
    x = np.linspace(0.0, 1.0, 256)
    name = name.lower()
    if name == "jet":
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    elif name == "inferno":
        r = np.clip(x * 1.5, 0.0, 1.0)
        g = np.clip(x * 1.2 - 0.2, 0.0, 1.0)
        b = np.clip(x * 0.8 - 0.4, 0.0, 1.0)
    else:
        # Turbo perceptualmente uniforme (aproximação suave)
        r = np.clip(np.sin(x * np.pi / 2.0), 0.0, 1.0)
        g = np.clip(np.sin(x * np.pi), 0.0, 1.0)
        b = np.clip(np.cos(x * np.pi / 2.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


class GradCAMExplainer:
    """Mecanismo de IA Explicável (XAI) para inspeção de atenção visual no System 1.

    Extrai mapas de calor espaciais através de Grad-CAM e Saliency Maps a partir
    da última camada convolucional do extrator visual IMPALA CNN (block3).

    Suporta 3 modos cognitivos de atenção:
      1. 'gradcam_policy': Relevância espacial que motivou a ação escolhida pela política.
      2. 'saliency_value': Regiões que elevam ou derrubam a expectativa de retorno do Crítico V(s).
      3. 'saliency_entropy': Regiões visuais responsáveis por gerar dúvida ou indecisão (entropia).
    """

    def __init__(self, agent: UniversalS1Agent, colormap: str = "turbo") -> None:
        self.agent = agent
        self.colormap_name = colormap
        self._lut = _build_colormap_lut(colormap)
        self._target_layer: Optional[nn.Module] = None
        self._activations: List[torch.Tensor] = []
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        if self.is_supported():
            self._target_layer = self.agent.front_end.visual_encoder.block3
            self._hook_handle = self._target_layer.register_forward_hook(self._forward_hook)

    def is_supported(self) -> bool:
        """Verifica se o agente possui codificador visual IMPALA ativo."""
        return (
            hasattr(self.agent, "is_visual")
            and bool(self.agent.is_visual)
            and hasattr(self.agent, "front_end")
            and hasattr(self.agent.front_end, "visual_encoder")
            and hasattr(self.agent.front_end.visual_encoder, "block3")
        )

    def _forward_hook(self, module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        self._activations.clear()
        self._activations.append(out)

    def compute_saliency(
        self,
        obs_dict: Dict[str, Any],
        hx: Optional[torch.Tensor] = None,
        mode: str = "gradcam_policy",
        action: Optional[Union[int, np.ndarray]] = None,
        action_name: Optional[str] = None,
        action_names: Optional[List[str]] = None,
    ) -> Tuple[Optional[np.ndarray], str]:
        """Calcula o mapa de calor normalizado [0, 1] e a etiqueta de texto explicativa.

        Args:
            obs_dict: Dicionário causal com 'obs', 'prev_action', 'prev_reward'.
            hx: Estado oculto recorrente do GRU.
            mode: 'gradcam_policy', 'saliency_value' ou 'saliency_entropy'.
            action: Ação predita para guiar o Grad-CAM.
            action_name: Descrição textual legível da ação (ex: 'MOVE_FORWARD + ATTACK').
            action_names: Lista ordenada opcional com os nomes de todas as ações disponíveis.

        Returns:
            Tuple de (heatmap_2d [11, 11] float32 ou None, label descritivo).
        """
        if not self.is_supported():
            return None, "MODO VETORIAL (Sem CNN)"

        device = self.agent.device
        was_training = self.agent.training
        self.agent.eval()

        # Prepara tensores com suporte a gradiente local
        obs = torch.as_tensor(obs_dict["obs"], dtype=torch.float32, device=device)
        prev_act = torch.as_tensor(obs_dict["prev_action"], device=device)
        prev_rew = torch.as_tensor(obs_dict["prev_reward"], dtype=torch.float32, device=device)

        if obs.dim() == 3:
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif obs.dim() == 4:
            obs = obs.unsqueeze(0)

        if prev_act.dim() == 0:
            prev_act = prev_act.view(1, 1)
        elif prev_act.dim() == 1:
            prev_act = prev_act.unsqueeze(0)

        if prev_rew.dim() == 0:
            prev_rew = prev_rew.view(1, 1, 1)
        elif prev_rew.dim() == 1:
            prev_rew = prev_rew.view(1, 1, 1)

        hx_detached = hx.detach().to(device) if hx is not None else None

        label = ""
        heatmap_norm = None

        with torch.enable_grad():
            self._activations.clear()
            z = self.agent.front_end(obs, prev_act, prev_rew)
            h_t, _ = self.agent.trunk(z, hx=hx_detached)
            dist = self.agent.policy_head(h_t)
            val = self.agent.value_head(h_t)

            if not self._activations:
                if was_training:
                    self.agent.train()
                return None, "SEM ATIVAÇÕES"

            act_feat = self._activations[0]  # [B, 32, 11, 11]

            if mode == "saliency_value":
                target_score = val[0, 0, 0]
                val_num = float(target_score.item())
                label = f"VALOR V(s): {val_num:+.1f}"
            elif mode == "saliency_entropy":
                entropy = dist.entropy()[0, 0]
                if entropy.dim() > 0:
                    entropy = entropy.sum()
                ent_num = float(entropy.item())
                target_score = entropy
                label = f"INCERTEZA H: {ent_num:.2f}"
            else:
                # 'gradcam_policy' padrão
                act_idx = 0
                if action is not None:
                    act_idx = int(action) if np.isscalar(action) else int(action[0])
                elif hasattr(dist, "logits"):
                    act_idx = int(torch.argmax(dist.logits[0, 0]).item())

                if action_name is None and action_names is not None and 0 <= act_idx < len(action_names):
                    action_name = action_names[act_idx]

                act_lbl = action_name or f"AÇÃO #{act_idx}"
                label = f"FOCO: {act_lbl}"

                if hasattr(dist, "logits"):
                    target_score = dist.logits[0, 0, act_idx]
                else:
                    target_score = dist.mean[0, 0, act_idx]

            try:
                grads = torch.autograd.grad(target_score, act_feat, retain_graph=False)[0]
                weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, 32, 1, 1]
                cam = torch.relu((weights * act_feat).sum(dim=1))  # [1, 11, 11]
                cam_np = cam.squeeze().detach().cpu().numpy()

                min_v, max_v = float(cam_np.min()), float(cam_np.max())
                if max_v - min_v > 1e-8:
                    heatmap_norm = (cam_np - min_v) / (max_v - min_v)
                else:
                    heatmap_norm = np.zeros_like(cam_np, dtype=np.float32)
            except Exception:
                heatmap_norm = None

        if was_training:
            self.agent.train()

        return heatmap_norm, label

    def render_overlay(
        self,
        frame_rgb: np.ndarray,
        heatmap_2d: Optional[np.ndarray],
        label: Optional[str] = None,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Sobrepõe o mapa de calor térmico e renderiza badge informativo sobre o frame original."""
        if frame_rgb is None or not isinstance(frame_rgb, np.ndarray):
            return frame_rgb

        arr = frame_rgb
        # Normalização de dimensões: CHW -> HWC se necessário
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        h, w = arr.shape[:2]

        if heatmap_2d is not None and heatmap_2d.ndim == 2:
            # Upsampling bilinear de alta velocidade do mapa 11x11 para a resolução do viewport
            cam_uint8 = (np.clip(heatmap_2d, 0.0, 1.0) * 255.0).astype(np.uint8)
            cam_img = Image.fromarray(cam_uint8).resize((w, h), resample=Image.BILINEAR)
            cam_upsampled = np.array(cam_img)

            # Aplicação do colormap vetorizado via tabela LUT
            heatmap_rgb = self._lut[cam_upsampled]

            # Alpha Blending suave
            blended = (
                arr.astype(np.float32) * (1.0 - alpha)
                + heatmap_rgb.astype(np.float32) * alpha
            ).astype(np.uint8)
        else:
            blended = arr.copy()

        # Renderização de HUD Badge no topo do frame
        if label:
            pil_img = Image.fromarray(blended)
            draw = ImageDraw.Draw(pil_img)

            # Caixa semi-transparente estilizada no canto superior esquerdo
            badge_text = f"👁️ {label}"
            text_w = len(badge_text) * 7 + 12
            draw.rectangle([8, 8, min(w - 8, 8 + text_w), 26], fill=(11, 16, 28), outline=(56, 189, 248), width=1)
            draw.text((14, 11), badge_text, fill=(56, 189, 248))

            return np.array(pil_img)

        return blended

    def close(self) -> None:
        """Remove hooks do PyTorch para liberação de memória."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
