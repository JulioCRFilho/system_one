import http.client
import json
import time
import numpy as np
import pytest

from system1_engine.hud.frame_buffer import VideoFrameBuffer
from system1_engine.hud.runner import HUDProcessRunner
from system1_engine.hud.server import HUDServer


def test_video_frame_buffer_placeholder_and_update():
    buf = VideoFrameBuffer(width=320, height=240)
    frame0 = buf.get_frame()
    assert isinstance(frame0, bytes)
    assert len(frame0) > 100
    # Check JPEG header
    assert frame0[:2] == b"\xff\xd8"

    dummy_jpeg = b"\xff\xd8\xff\xe0testjpeg"
    buf.update_frame(dummy_jpeg)
    assert buf.get_frame() == dummy_jpeg

    buf.reset_placeholder("TEST TITLE", "TEST SUBTITLE")
    frame1 = buf.get_frame()
    assert frame1[:2] == b"\xff\xd8"


def test_hud_server_lifecycle_and_endpoints():
    server = HUDServer(host="127.0.0.1", port=8995, open_browser=False)
    with server:
        time.sleep(0.1)
        conn = http.client.HTTPConnection("127.0.0.1", 8995, timeout=2.0)

        # 1. Test GET /
        conn.request("GET", "/")
        res = conn.getresponse()
        assert res.status == 200
        html = res.read().decode("utf-8")
        assert "SYSTEM 1 ENGINE" in html
        assert "Visual Control HUD" in html

        # 2. Test GET /api/state
        conn.request("GET", "/api/state")
        res = conn.getresponse()
        assert res.status == 200
        state = json.loads(res.read().decode("utf-8"))
        assert "runner" in state
        assert "checkpoints" in state
        assert "telemetry" in state
        assert state["runner"]["status"] == "IDLE"

        # 3. Test POST /api/internal/telemetry
        telemetry_payload = json.dumps({"fps": 60.5, "total_steps": 1234}).encode("utf-8")
        conn.request(
            "POST",
            "/api/internal/telemetry",
            telemetry_payload,
            {"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 200
        res.read()
        snap = server.get_telemetry_snapshot()
        assert snap["fps"] == 60.5
        assert snap["total_steps"] == 1234

        # 4. Test POST /api/internal/frame
        dummy_frame = b"\xff\xd8\xff\xe0dummydata"
        conn.request(
            "POST",
            "/api/internal/frame",
            dummy_frame,
            {"Content-Type": "image/jpeg"},
        )
        res = conn.getresponse()
        assert res.status == 200
        res.read()
        assert server.frame_buffer.get_frame() == dummy_frame

        # 5. Test POST /api/action stop
        action_payload = json.dumps({"action": "stop"}).encode("utf-8")
        conn.request("POST", "/api/action", action_payload, {"Content-Type": "application/json"})
        res = conn.getresponse()
        assert res.status == 200
        action_resp = json.loads(res.read().decode("utf-8"))
        assert action_resp["success"] is True

        conn.close()


def test_hud_process_runner_start_and_stop():
    runner = HUDProcessRunner()
    assert not runner.is_running()
    assert runner.get_state()["status"] == "IDLE"

    # Start a benchmark task with few steps
    config = {
        "mode": "benchmark",
        "steps": 100,
    }
    success, msg = runner.start(config, "http://127.0.0.1:8995")
    assert success is True
    assert runner.is_running()

    # Wait briefly and verify logs appear
    time.sleep(0.4)
    logs, count = runner.get_logs(0)
    assert count > 0

    # Stop runner
    stop_ok, stop_msg = runner.stop()
    assert stop_ok is True
    assert not runner.is_running()
    assert runner.get_state()["status"] in ["STOPPED", "COMPLETED"]


def test_hud_e2e_evaluation_and_in_browser_rendering():
    server = HUDServer(host="127.0.0.1", port=8996, open_browser=False)
    with server:
        config = {
            "mode": "run",
            "env": "CartPole-v1",
            "episodes": 1,
            "fps": 0.0,
            "render_mode": "in_browser",
        }
        success, msg = server.runner.start(config, "http://127.0.0.1:8996")
        assert success is True

        t0 = time.time()
        while server.runner.is_running() and (time.time() - t0 < 8.0):
            time.sleep(0.1)

        state = server.runner.get_state()
        assert state["status"] == "COMPLETED"
        assert not server.runner.is_running()

        # Verifica se frames de vídeo foram transmitidos para o buffer MJPEG
        current_frame = server.frame_buffer.get_frame()
        assert current_frame[:2] == b"\xff\xd8"
        assert current_frame != server.frame_buffer._default_frame

        # Verifica se telemetria foi atualizada
        snap = server.get_telemetry_snapshot()
        assert snap["total_steps"] > 0

        # Verifica logs gerados
        logs, count = server.runner.get_logs(0)
        assert count > 0
        assert any("Episódio 1/1 finalizado" in line for line in logs)

