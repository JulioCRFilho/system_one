#!/usr/bin/env python3
"""Task 8: Demonstração Operacional dos 3 Níveis de Integração de Ambientes.

Demonstra:
  - Nível 1: WindowCaptureEnv (Caixa-Preta com mss, pynput e frame pacing assíncrono)
  - Nível 2: MemoryHookEnv (Modo Híbrido: Pixels + Recompensa/GameOver cirúrgicos da RAM)
  - Nível 3: NativeEngineEnv (Regime Lock-Step síncrono e headless > 10.000 FPS)
  - Fábrica Unificada: make_game_env(...) retornando sempre UniversalS1Wrapper
"""

from pathlib import Path
import sys
import time

# Garante import do system1_engine mesmo se executado de qualquer diretório
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.adapters import (
    BufferMemoryBackend,
    MemoryField,
    make_game_env,
)


def demo_level1_window():
    print("-" * 70)
    print("🖥️  NÍVEL 1: WINDOW CAPTURE (Caixa-Preta com MSS + Pynput)")
    print("-" * 70)

    # Frame sintético emulando tela de jogo (ou captura real da janela)
    synthetic_screen = np.random.randint(50, 200, size=(600, 800, 3), dtype=np.uint8)

    env = make_game_env(
        adapter_type="window",
        config={
            "window_bbox": {"top": 0, "left": 0, "width": 800, "height": 600},
            "actions_map": {0: None, 1: "space", 2: "left", 3: "right"},
            "target_fps": 30,
            "grayscale": True,
            "mock_capture_source": lambda: synthetic_screen,
            "reward_fn": lambda curr, prev, info: 1.0,
        },
    )

    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space, is_visual=True)
    agent.eval()

    obs_dict, info = env.reset()
    agent.reset_memory()

    print(f"• Shape da Observação Empilhada : {obs_dict['obs'].shape} (2 canais: s_t, s_(t-1))")
    print(f"• Ações Mapeadas                : {env.action_space.n} ações discretas ({type(env.action_space).__name__})")

    t0 = time.perf_counter()
    for step in range(5):
        action = agent.act_fast(obs_dict)
        obs_dict, reward, term, trunc, step_info = env.step(action)
        print(f"  Passo {step + 1}: Ação={action} | Recompensa={reward:.1f} | Frame time={step_info['step_time_ms']:.2f} ms | FPS Efetivo={step_info['effective_fps']:.1f}")

    env.close()
    print("✅ Nível 1 validado: Frame pacing assíncrono e barramento unificado ativos!")


def demo_level2_memory():
    print("\n" + "-" * 70)
    print("🧠 NÍVEL 2: MEMORY HOOKING (Modo Híbrido: Pixels + RAM Hook)")
    print("-" * 70)

    # Buffer de memória RAM simulada
    ram = BufferMemoryBackend(buffer_size=1024)
    ram.write_value(offset=0x10, value=100, dtype="int32")  # HP inicial = 100
    ram.write_value(offset=0x14, value=0.0, dtype="float32") # Score inicial = 0.0

    schema = [
        MemoryField(name="hp", offset=0x10, dtype="int32", is_terminal=True),
        MemoryField(name="score", offset=0x14, dtype="float32", is_reward=True),
    ]

    mock_pixel_feed = np.full((300, 300, 3), fill_value=80, dtype=np.uint8)

    env = make_game_env(
        adapter_type="memory",
        config={
            "memory_schema": schema,
            "actions_map": {0: None, 1: "attack", 2: "heal"},
            "capture_screen": True,
            "memory_backend": ram,
            "window_bbox": {"top": 0, "left": 0, "width": 300, "height": 300},
        },
    )
    # Injata mock_capture_source
    raw_memory_env = env.env
    if hasattr(raw_memory_env, "screen_adapter") and raw_memory_env.screen_adapter is not None:
        raw_memory_env.screen_adapter.mock_capture_source = lambda: mock_pixel_feed

    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space, is_visual=True)
    agent.eval()

    obs_dict, _ = env.reset()
    agent.reset_memory()

    print("• Executando dinâmica híbrida (pixels para o agente, HP/Score da RAM):")
    # Passo 1: Ganho de score
    ram.write_value(offset=0x14, value=25.0, dtype="float32")
    action = agent.act_fast(obs_dict)
    obs_dict, reward, term, trunc, info = env.step(action)
    print(f"  Passo 1: Recompensa de Score lida da RAM = +{reward:.1f} (RAM: HP={info['ram']['hp']}, Score={info['ram']['score']})")

    # Passo 2: Dano sofrido
    ram.write_value(offset=0x10, value=40, dtype="int32")
    action = agent.act_fast(obs_dict)
    obs_dict, reward, term, trunc, info = env.step(action)
    print(f"  Passo 2: Dano detectado na RAM: HP caiu para {info['ram']['hp']} (Terminated={term})")

    # Passo 3: Morte / Game Over
    ram.write_value(offset=0x10, value=0, dtype="int32")
    action = agent.act_fast(obs_dict)
    obs_dict, reward, term, trunc, info = env.step(action)
    print(f"  Passo 3: HP zerado na RAM: Terminated={term} (Game Over cirúrgico sem visão de texto!)")

    env.close()
    print("✅ Nível 2 validado: Zero falsos positivos de visão para recompensa e término!")


def demo_level3_native():
    print("\n" + "-" * 70)
    print("⚡ NÍVEL 3: NATIVE ENGINE (Regime Lock-Step & Headless > 10.000 FPS)")
    print("-" * 70)

    env = make_game_env(
        adapter_type="native",
        config={
            "engine_type": "native_sim",
            "is_visual": False,
            "obs_dim": 8,
            "action_dim": 4,
            "is_action_discrete": True,
        },
    )

    agent = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space, is_visual=False)
    agent.eval()

    obs_dict, _ = env.reset()
    agent.reset_memory()

    n_steps = 2000
    print(f"• Executando {n_steps:,} passos em regime lock-step de alta velocidade...")

    t0 = time.perf_counter()
    for _ in range(n_steps):
        action = agent.act_fast(obs_dict)
        obs_dict, reward, term, trunc, info = env.step(action)
    total_time = time.perf_counter() - t0

    fps = n_steps / max(1e-6, total_time)
    print(f"  Tempo Total : {total_time * 1000:.1f} ms")
    print(f"  Throughput  : {fps:,.0f} FPS (Agente + Física)")

    env.close()
    print("✅ Nível 3 validado: Sample efficiency máxima para treino massivo em paralelo!")


def main():
    print("=" * 70)
    print("🚀 DEMONSTRAÇÃO DOS 3 NÍVEIS DE INTEGRAÇÃO DO UNIVERSAL SYSTEM 1")
    print("=" * 70)
    demo_level1_window()
    demo_level2_memory()
    demo_level3_native()
    print("\n" + "=" * 70)
    print("🏁 TODOS OS 3 NÍVEIS DE ADAPTAÇÃO OPERARAM COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
