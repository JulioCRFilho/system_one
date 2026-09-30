import argparse
import http.client
import io
import json
import sys
import time
import urllib.parse
from typing import Any, Dict, Optional
import gymnasium as gym
import numpy as np
from PIL import Image
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.adapters import make_game_env
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.telemetry.tracker import LiveStatsTracker
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.transfer.manager import KnowledgeTransferManager


class HUDClient:
    """Cliente HTTP interno de alta performance e baixa latência para comunicação com o HUDServer."""

    def __init__(self, server_url: str) -> None:
        parsed = urllib.parse.urlparse(server_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 8050
        self._conn: Optional[http.client.HTTPConnection] = None

    def _get_connection(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self.host, self.port, timeout=2.0)
        return self._conn

    def send_telemetry(self, data: Dict[str, Any]) -> None:
        payload = json.dumps(data).encode("utf-8")
        headers = {"Content-Type": "application/json", "Content-Length": str(len(payload))}
        for attempt in range(2):
            try:
                conn = self._get_connection()
                conn.request("POST", "/api/internal/telemetry", payload, headers)
                res = conn.getresponse()
                res.read()
                return
            except Exception:
                self._conn = None

    def send_frame(self, frame: np.ndarray) -> None:
        """Codifica array RGB em JPEG de alta performance e envia via POST binário."""
        try:
            arr = frame
            # Normalização de dimensões: CHW -> HWC se necessário
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)

            img = Image.fromarray(arr)
            # Redimensiona se for excessivamente grande para manter latência < 2ms
            if img.width > 640 or img.height > 480:
                img.thumbnail((640, 480))

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            jpeg_bytes = buf.getvalue()

            headers = {"Content-Type": "image/jpeg", "Content-Length": str(len(jpeg_bytes))}
            for attempt in range(2):
                try:
                    conn = self._get_connection()
                    conn.request("POST", "/api/internal/frame", jpeg_bytes, headers)
                    res = conn.getresponse()
                    res.read()
                    return
                except Exception:
                    self._conn = None
        except Exception:
            pass

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


def build_hud_env(
    env_id: str,
    scenario: Optional[str] = None,
    render_mode: str = "in_browser",
) -> UniversalS1Wrapper:
    """Instancia o ambiente com suporte a renderização in-browser (rgb_array), janela ou headless."""
    if env_id.lower() in ["vizdoom", "native"]:
        scenario_path = scenario or "basic.cfg"
        headless = render_mode != "window"
        return make_game_env(
            "native",
            {
                "engine_type": "vizdoom",
                "scenario_path": scenario_path,
                "args": {"headless": headless},
            },
        )

    # Gym environments
    gym_render_mode = None
    if render_mode == "in_browser":
        gym_render_mode = "rgb_array"
    elif render_mode == "window":
        gym_render_mode = "human"

    raw_env = gym.make(env_id, render_mode=gym_render_mode)
    return UniversalS1Wrapper(raw_env)


def run_worker_train(config: Dict[str, Any], hud_client: HUDClient) -> None:
    env_id = config.get("env", "CartPole-v1")
    render_mode = config.get("render_mode", "in_browser")
    steps = int(config.get("steps", 40000))
    lr = float(config.get("lr", 7e-4))
    entropy_coef = float(config.get("entropy_coef", 0.005))
    target_return = float(config.get("target_return", 475.0))
    rollout_steps = int(config.get("rollout_steps", 1024))
    chunk_length = int(config.get("chunk_length", 16))
    chunk_batch_size = int(config.get("chunk_batch_size", 16))
    save_path = config.get("save")
    transfer_from = config.get("transfer_from")
    freeze_trunk = bool(config.get("freeze_trunk", True))

    print(f"=== [HUD Worker] Modo de Treino Iniciado no Ambiente: {env_id} ===")
    print(f"Configuração: steps={steps}, lr={lr}, entropy_coef={entropy_coef}, target_return={target_return}")
    if transfer_from:
        print(f"Transferência de pesos ativa a partir de: {transfer_from} (freeze_trunk={freeze_trunk})")

    env = build_hud_env(env_id, scenario=config.get("scenario"), render_mode=render_mode)
    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
    )

    if transfer_from:
        loaded = KnowledgeTransferManager.load_transferable_weights(
            agent=agent,
            checkpoint_path=transfer_from,
            freeze_trunk=freeze_trunk,
        )
        print(f"Trunk transferido com sucesso: {len(loaded)} parâmetros carregados.")

    tracker = LiveStatsTracker()
    last_telemetry_ts = 0.0
    last_frame_ts = 0.0

    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        learning_rate=lr,
        rollout_steps=rollout_steps,
        chunk_length=chunk_length,
        chunk_batch_size=chunk_batch_size,
        entropy_coef=entropy_coef,
        tracker=tracker,
    )

    def train_callback(cur_steps: int, mean_ret: float) -> None:
        nonlocal last_telemetry_ts, last_frame_ts
        now = time.time()
        # Telemetria a cada ~100 ms
        if now - last_telemetry_ts >= 0.1:
            snap = tracker.snapshot()
            snap["grad_norms"] = tracker.metrics.grad_norms
            hud_client.send_telemetry(snap)
            last_telemetry_ts = now

        # Renderização in-browser durante treino (amostrada a ~20 FPS para não atrasar o PPO)
        if render_mode == "in_browser" and (now - last_frame_ts >= 0.05):
            try:
                frame = env.render()
                if frame is not None:
                    hud_client.send_frame(frame)
                    last_frame_ts = now
            except Exception:
                pass

    try:
        final_return = trainer.train(
            max_steps=steps,
            target_return=target_return,
            callback=train_callback,
            verbose=True,
        )

        if save_path:
            KnowledgeTransferManager.save_checkpoint(
                agent=agent,
                checkpoint_path=save_path,
                extra_info={"env_id": env_id, "final_return": final_return, "steps": trainer.total_steps},
            )
            print(f"Checkpoint salvo com sucesso em: {save_path}")

        tracker.set_completed(True)
        final_snap = tracker.snapshot()
        final_snap["grad_norms"] = tracker.metrics.grad_norms
        hud_client.send_telemetry(final_snap)
        print(f"=== [HUD Worker] Treinamento Concluído! Retorno Final: {final_return:.2f} ===")

    finally:
        env.close()


def run_worker_eval(config: Dict[str, Any], hud_client: HUDClient) -> None:
    env_id = config.get("env", "CartPole-v1")
    render_mode = config.get("render_mode", "in_browser")
    episodes = int(config.get("episodes", 5))
    load_path = config.get("load")
    save_path = config.get("save")
    use_fast = config.get("inference_mode") == "fast"
    fps_target = float(config.get("fps", 50.0))
    dt_target = (1.0 / fps_target) if fps_target > 0 else 0.0

    mode_desc = "act_fast() [Reflexo Puro]" if use_fast else "act_with_confidence() [Gating]"
    print(f"=== [HUD Worker] Modo de Avaliação Iniciado no Ambiente: {env_id} ===")
    print(f"Episódios: {episodes} | Decisão: {mode_desc} | FPS Alvo: {fps_target} | Renderização: {render_mode}")

    env = build_hud_env(env_id, scenario=config.get("scenario"), render_mode=render_mode)
    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
    )

    if load_path:
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        agent.load_state_dict(state_dict, strict=False)
        print(f"Pesos carregados com sucesso de: {load_path}")
    else:
        print("[Aviso] Nenhum checkpoint especificado; utilizando pesos aleatórios inicializados.")

    agent.eval()
    tracker = LiveStatsTracker()
    last_telemetry_ts = 0.0

    try:
        for ep in range(episodes):
            obs_dict, _ = env.reset()
            agent.reset_memory()
            ep_reward = 0.0
            steps = 0
            done = False

            while not done:
                t_step_start = time.perf_counter()

                t0 = time.perf_counter_ns()
                if use_fast:
                    action = agent.act_fast(obs_dict)
                    lat_us = (time.perf_counter_ns() - t0) / 1000.0
                    tracker.record_inference(
                        latency_us=lat_us,
                        uncertainty=0.0,
                        confidence=1.0,
                        entropy=0.0,
                    )
                else:
                    decision = agent.act_with_confidence(obs_dict)
                    lat_us = (time.perf_counter_ns() - t0) / 1000.0
                    action = decision.action
                    tracker.record_inference(
                        latency_us=lat_us,
                        uncertainty=decision.uncertainty,
                        confidence=decision.confidence,
                        entropy=decision.entropy,
                    )

                obs_dict, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                steps += 1
                tracker.record_env_step(reward=reward, done=done)

                # Renderização in-browser
                if render_mode == "in_browser":
                    try:
                        frame = env.render()
                        if frame is not None:
                            hud_client.send_frame(frame)
                    except Exception:
                        pass

                # Atualização periódica de telemetria a cada ~50 ms
                now = time.time()
                if now - last_telemetry_ts >= 0.05:
                    snap = tracker.snapshot()
                    hud_client.send_telemetry(snap)
                    last_telemetry_ts = now

                # Pacing da simulação para suavidade visual no HUD
                if dt_target > 0:
                    elapsed = time.perf_counter() - t_step_start
                    sleep_time = dt_target - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            print(f"Episódio {ep + 1}/{episodes} finalizado | Retorno: {ep_reward:.1f} | Passos: {steps}")

        if save_path:
            KnowledgeTransferManager.save_checkpoint(
                agent=agent,
                checkpoint_path=save_path,
                extra_info={"env_id": env_id, "evaluated_episodes": episodes},
            )
            print(f"Checkpoint salvo com sucesso em: {save_path}")

        tracker.set_completed(True)
        hud_client.send_telemetry(tracker.snapshot())
        print("=== [HUD Worker] Avaliação Concluída com Sucesso! ===")

    finally:
        env.close()


def run_worker_benchmark(config: Dict[str, Any], hud_client: HUDClient) -> None:
    steps = int(config.get("steps", 5000))
    print(f"=== [HUD Worker] Iniciando Benchmark de Latência CPU ({steps} passos) ===")
    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)
    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
    )
    agent.eval()
    obs_dict, _ = env.reset()
    agent.reset_memory()

    # Warmup
    for _ in range(50):
        agent.act_fast(obs_dict)

    latencies = []
    tracker = LiveStatsTracker()

    t_start = time.time()
    for s in range(steps):
        t0 = time.perf_counter_ns()
        agent.act_fast(obs_dict)
        t1 = time.perf_counter_ns()
        lat_us = (t1 - t0) / 1000.0
        latencies.append(lat_us / 1000.0)  # ms
        tracker.record_inference(latency_us=lat_us, uncertainty=0.01, confidence=0.99, entropy=0.0)

        if s % 500 == 0:
            now = time.time()
            snap = tracker.snapshot()
            hud_client.send_telemetry(snap)

    avg_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    p99_ms = float(np.percentile(latencies, 99))

    print(f"Resultado do Benchmark ({steps} passos):")
    print(f"  Latência Média: {avg_ms:.4f} ms ({avg_ms * 1000:.1f} µs)")
    print(f"  P95: {p95_ms:.4f} ms | P99: {p99_ms:.4f} ms")
    status_str = "APROVADO (<= 0.80 ms)" if avg_ms <= 0.8 else "REPROVADO (> 0.80 ms)"
    print(f"  Conformidade com o Orçamento: {status_str}")

    tracker.set_completed(True)
    hud_client.send_telemetry(tracker.snapshot())
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="HUD Execution Worker")
    parser.add_argument("--config", type=str, required=True, help="JSON configuration payload")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8050", help="HUD server URL")
    args = parser.parse_args()

    config = json.loads(args.config)
    mode = config.get("mode", "train")

    hud_client = HUDClient(args.server_url)

    try:
        if mode == "train":
            run_worker_train(config, hud_client)
        elif mode == "run":
            run_worker_eval(config, hud_client)
        elif mode == "benchmark":
            run_worker_benchmark(config, hud_client)
        else:
            print(f"[Erro] Modo desconhecido: '{mode}'")
            sys.exit(1)
    finally:
        hud_client.close()


if __name__ == "__main__":
    main()
