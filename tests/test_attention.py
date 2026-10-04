import http.client
import json
import time
import gymnasium as gym
import numpy as np
import pytest
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.core.attention import GradCAMExplainer, _build_colormap_lut
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.hud.server import HUDServer
from system1_engine.hud.worker import HUDClient


def test_colormap_lut_generation():
    """Valida a geração das tabelas de consulta de cores RGB (LUT 256x3)."""
    for cmap in ["turbo", "jet", "inferno"]:
        lut = _build_colormap_lut(cmap)
        assert isinstance(lut, np.ndarray)
        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8
        assert lut.min() >= 0
        assert lut.max() <= 255


def test_gradcam_explainer_support_detection():
    """Valida se o explainer detecta corretamente agentes com IMPALA CNN vs agentes vetoriais."""
    # Agente visual (CNN IMPALA ativa) com 2*C canais empilhados
    vis_obs_space = gym.spaces.Box(0.0, 1.0, shape=(2, 84, 84), dtype=np.float32)
    vis_act_space = gym.spaces.Discrete(3)
    vis_agent = UniversalS1Agent(vis_obs_space, vis_act_space)
    explainer_vis = GradCAMExplainer(vis_agent)
    assert explainer_vis.is_supported() is True
    explainer_vis.close()

    # Agente puramente vetorial (sem CNN)
    vec_obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
    vec_act_space = gym.spaces.Discrete(2)
    vec_agent = UniversalS1Agent(vec_obs_space, vec_act_space)
    explainer_vec = GradCAMExplainer(vec_agent)
    assert explainer_vec.is_supported() is False
    heatmap, label = explainer_vec.compute_saliency({"obs": np.zeros(4, dtype=np.float32)})
    assert heatmap is None
    assert "MODO VETORIAL" in label
    explainer_vec.close()


def test_gradcam_cognitive_modes_and_action_names():
    """Valida o cálculo dos 3 modos cognitivos: política, valor e entropia com nomes de ações."""
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(2, 84, 84), dtype=np.float32)
    act_space = gym.spaces.Discrete(4)
    agent = UniversalS1Agent(obs_space, act_space)
    explainer = GradCAMExplainer(agent, colormap="turbo")

    # Dicionário de observação causal com stacking visual
    obs_dict = {
        "obs": np.random.rand(2, 84, 84).astype(np.float32),
        "prev_action": 1,
        "prev_reward": 0.5,
    }
    action_names = ["GIRAR_ESQ", "GIRAR_DIR", "AVANÇAR", "DISPARAR"]

    # Warmup
    explainer.compute_saliency(obs_dict=obs_dict, mode="gradcam_policy", action=0)

    # 1. Modo 'gradcam_policy' com ação forçada e nome
    t0 = time.perf_counter()
    heatmap_p, label_p = explainer.compute_saliency(
        obs_dict=obs_dict,
        mode="gradcam_policy",
        action=3,
        action_names=action_names,
    )
    lat_ms = (time.perf_counter() - t0) * 1000.0

    assert heatmap_p is not None
    assert heatmap_p.shape == (11, 11)
    assert heatmap_p.min() >= 0.0
    assert heatmap_p.max() <= 1.0
    assert "FOCO: DISPARAR" in label_p
    assert lat_ms < 10.0, f"Grad-CAM demorou {lat_ms:.2f} ms (limite: 10 ms)"

    # 2. Modo 'gradcam_policy' inferindo argmax automaticamente
    heatmap_auto, label_auto = explainer.compute_saliency(
        obs_dict=obs_dict,
        mode="gradcam_policy",
        action=None,
        action_names=action_names,
    )
    assert heatmap_auto is not None
    assert any(name in label_auto for name in action_names)

    # 3. Modo 'saliency_value' (Crítico V(s))
    heatmap_v, label_v = explainer.compute_saliency(
        obs_dict=obs_dict,
        mode="saliency_value",
    )
    assert heatmap_v is not None
    assert heatmap_v.shape == (11, 11)
    assert "VALOR V(s):" in label_v

    # 4. Modo 'saliency_entropy' (Incerteza H)
    heatmap_e, label_e = explainer.compute_saliency(
        obs_dict=obs_dict,
        mode="saliency_entropy",
    )
    assert heatmap_e is not None
    assert heatmap_e.shape == (11, 11)
    assert "INCERTEZA H:" in label_e

    explainer.close()


def test_render_overlay_and_badge():
    """Valida a renderização do mapa térmico e do badge com PIL e NumPy puro."""
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(2, 84, 84), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)
    agent = UniversalS1Agent(obs_space, act_space)
    explainer = GradCAMExplainer(agent)

    frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    heatmap = np.random.rand(11, 11).astype(np.float32)

    rendered = explainer.render_overlay(frame, heatmap, label="FOCO: DISPARAR", alpha=0.45)
    assert isinstance(rendered, np.ndarray)
    assert rendered.shape == (240, 320, 3)
    assert rendered.dtype == np.uint8
    # Frame foi modificado pelo heatmap e badge
    assert not np.array_equal(frame, rendered)

    # Teste de robustez com heatmap None
    no_heat = explainer.render_overlay(frame, None, label="MODO NORMAL")
    assert no_heat.shape == (240, 320, 3)

    explainer.close()


def test_hud_server_vision_mode_api_and_client_sync():
    """Valida o endpoint /api/action de alternância de modo de visão e sincronização com HUDClient."""
    server = HUDServer(host="127.0.0.1", port=8994, open_browser=False)
    with server:
        time.sleep(0.1)
        conn = http.client.HTTPConnection("127.0.0.1", 8994, timeout=2.0)

        # 1. Verifica estado inicial em /api/state
        conn.request("GET", "/api/state")
        res = conn.getresponse()
        assert res.status == 200
        state = json.loads(res.read().decode("utf-8"))
        assert state.get("vision_mode") == "normal"

        # 2. Alterna modo via POST /api/action
        action_payload = json.dumps({"action": "set_vision_mode", "mode": "gradcam_policy"}).encode("utf-8")
        conn.request("POST", "/api/action", action_payload, {"Content-Type": "application/json"})
        res = conn.getresponse()
        assert res.status == 200
        action_resp = json.loads(res.read().decode("utf-8"))
        assert action_resp["success"] is True
        assert action_resp["vision_mode"] == "gradcam_policy"

        # 3. Verifica se /api/state reflete o novo modo
        conn.request("GET", "/api/state")
        res = conn.getresponse()
        state = json.loads(res.read().decode("utf-8"))
        assert state.get("vision_mode") == "gradcam_policy"

        # 4. Testa sincronização bidirecional transparente via HUDClient
        hud_client = HUDClient("http://127.0.0.1:8994", initial_vision_mode="normal")
        try:
            # Envia telemetria e verifica se a resposta do servidor atualiza o cliente
            hud_client.send_telemetry({"fps": 60.0})
            time.sleep(0.15)
            assert hud_client.get_vision_mode() == "gradcam_policy"

            # Alterna para saliency_value no servidor
            server.vision_mode = "saliency_value"
            dummy_frame = (np.zeros((100, 100, 3))).astype(np.uint8)
            hud_client.send_frame(dummy_frame)
            time.sleep(0.15)
            assert hud_client.get_vision_mode() == "saliency_value"
        finally:
            hud_client.close()

        conn.close()


def test_hud_e2e_vizdoom_attention_overlay():
    """Testa execução ponta a ponta do ViZDoom no worker com Grad-CAM ativo e overlay gerado."""
    server = HUDServer(host="127.0.0.1", port=8993, open_browser=False)
    server.vision_mode = "gradcam_policy"
    with server:
        config = {
            "mode": "run",
            "env": "vizdoom",
            "scenario": "basic.cfg",
            "episodes": 1,
            "fps": 0.0,
            "render_mode": "in_browser",
            "inference_mode": "confidence",
            "vision_mode": "gradcam_policy",
        }
        success, msg = server.runner.start(config, "http://127.0.0.1:8993")
        assert success is True

        t0 = time.time()
        while server.runner.is_running() and (time.time() - t0 < 10.0):
            time.sleep(0.1)

        state = server.runner.get_state()
        assert state["status"] == "COMPLETED"
        assert not server.runner.is_running()

        # Verifica se frames foram capturados e transmitidos com o badge do Grad-CAM
        frame = server.frame_buffer.get_frame()
        assert frame[:2] == b"\xff\xd8"
        assert frame != server.frame_buffer._default_frame
