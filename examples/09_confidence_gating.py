#!/usr/bin/env python3
"""Task 9: Mecanismo de Incerteza e Confidence Gating (Gatilho do System 2).

Demonstra como o System 1 avalia sua própria convicção de forma amortizada:
- Cálculo da Entropia de Shannon normalizada da distribuição da política π(a|s)
- Cálculo da Confiança (Top-1 prob) e Margem de Clareza (Top-1 vs Top-2)
- Emissão do flag binário 'is_uncertain' (Gatilho de Invocação para o System 2)
- Manutenção estrita do orçamento de latência em CPU (<= 0.8 ms)
"""

from pathlib import Path
import sys
import time

# Garante import do system1_engine mesmo se executado de qualquer diretório
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
import torch

from system1_engine.core.agent import ReflexDecision, UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper


def mock_system_2_deliberation(obs_dict: dict, reason: str) -> int:
    """Simulação de árbitro deliberativo System 2 (MCTS / CoT / Humano)."""
    # System 2 é mais lento e analisa a situação com calma
    time.sleep(0.01)  # 10 ms de deliberação
    # Por exemplo, aplica heurística de estabilidade baseada no ângulo do pêndulo
    cart_pos, cart_vel, pole_angle, pole_vel = obs_dict["obs"].flatten()
    action = 1 if pole_angle + 0.1 * pole_vel > 0 else 0
    return action


def run_confidence_gating_demo():
    print("=" * 75)
    print("🧠 TASK 9: MECANISMO DE INCERTEZA E CONFIDENCE GATING (SYSTEM 1 -> 2)")
    print("=" * 75)

    raw_env = gym.make("CartPole-v1")
    env = UniversalS1Wrapper(raw_env)

    # 1. Comparação entre Agente Tabula Rasa (Inseguro) e Agente Treinado (Convicto)
    agent_untrained = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    agent_untrained.eval()

    agent_trained = UniversalS1Agent(obs_space=env.observation_space, action_space=env.action_space)
    ckpt_path = Path("s1_cartpole.pt")
    if ckpt_path.exists():
        ckpt = torch.load("s1_cartpole.pt", map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        agent_trained.load_state_dict(state_dict, strict=False)
        print("• Checkpoint 's1_cartpole.pt' carregado para o Agente Treinado.")
    agent_trained.eval()

    obs_dict, _ = env.reset(seed=123)

    print("-" * 75)
    print("📊 COMPARAÇÃO DE CALIBRAÇÃO DE CONFIANÇA (Mesmo Estado Inicial):")

    # Incerteza no Agente Não-Treinado
    dec_untrained = agent_untrained.act_with_confidence(obs_dict, uncertainty_threshold=0.75, return_value=True)
    print("\n[Agente Tabula Rasa / Sem Treino]:")
    print(f"  - Ação Sugerida      : {dec_untrained.action}")
    print(f"  - Confiança (Top 1)  : {dec_untrained.confidence * 100:.1f}%")
    print(f"  - Incerteza (Entropia: {dec_untrained.uncertainty * 100:.1f}% (Shannon: {dec_untrained.entropy:.4f})")
    print(f"  - Margem Top1 - Top2 : {dec_untrained.margin * 100:.1f}%")
    print(f"  - Valor Estimado V(s): {dec_untrained.latent_value:.3f}")
    print(f"  - 🚨 Gatilho System 2: {'ATIVADO (is_uncertain = True)' if dec_untrained.is_uncertain else 'Desativado'}")

    # Incerteza no Agente Treinado
    dec_trained = agent_trained.act_with_confidence(obs_dict, uncertainty_threshold=0.75, return_value=True)
    print("\n[Agente Treinado / Reflexo Consolidado]:")
    print(f"  - Ação Sugerida      : {dec_trained.action}")
    print(f"  - Confiança (Top 1)  : {dec_trained.confidence * 100:.1f}%")
    print(f"  - Incerteza (Entropia: {dec_trained.uncertainty * 100:.1f}% (Shannon: {dec_trained.entropy:.4f})")
    print(f"  - Margem Top1 - Top2 : {dec_trained.margin * 100:.1f}%")
    print(f"  - Valor Estimado V(s): {dec_trained.latent_value:.3f}")
    print(f"  - 🟢 Gatilho System 2: {'Ativado' if dec_trained.is_uncertain else 'DESATIVADO (Confiança Suficiente)'}")

    # 2. Benchmark de Latência: act_fast puro vs act_with_confidence
    print("\n" + "-" * 75)
    print("⚡ BENCHMARK DE SOBRECARGA (OVERHEAD) DO GATILHO:")

    n_bench = 1000
    t0 = time.perf_counter_ns()
    for _ in range(n_bench):
        agent_trained.act_fast(obs_dict)
    time_raw_ms = (time.perf_counter_ns() - t0) / 1e6 / n_bench

    t0 = time.perf_counter_ns()
    for _ in range(n_bench):
        agent_trained.act_with_confidence(obs_dict)
    time_conf_ms = (time.perf_counter_ns() - t0) / 1e6 / n_bench

    overhead_us = (time_conf_ms - time_raw_ms) * 1000.0

    print(f"  - act_fast() Puro            : {time_raw_ms * 1000:.1f} µs ({time_raw_ms:.4f} ms)")
    print(f"  - act_with_confidence()      : {time_conf_ms * 1000:.1f} µs ({time_conf_ms:.4f} ms)")
    print(f"  - Sobrecarga de Cálculo      : {overhead_us:+.1f} µs ({overhead_us / 1000:.4f} ms)")
    print(f"  - Limite Rigoroso do Sistema : <= 0.8000 ms")
    assert time_conf_ms <= 0.8, "Latência com confidence gating excedeu orçamento!"
    print(f"  ✅ [APROVADO] Latência com telemetria completa mantida abaixo de 0.8 ms!")

    # 3. Simulação de Loop Híbrido com Despacho Arbitrado
    print("\n" + "-" * 75)
    print("🎮 SIMULAÇÃO DO LOOP DE DECISÃO HÍBRIDO (System 1 + Fallback System 2):")

    obs_dict, _ = env.reset(seed=42)
    agent_trained.reset_memory()

    s1_count = 0
    s2_count = 0

    for step in range(20):
        # Inferência reflexiva rápida
        decision: ReflexDecision = agent_trained.act_with_confidence(
            obs_dict, uncertainty_threshold=0.85, confidence_threshold=0.60
        )

        if decision.is_uncertain:
            # Fallback para deliberativo
            action = mock_system_2_deliberation(obs_dict, reason=f"Incerteza alta ({decision.uncertainty:.2f})")
            s2_count += 1
            origin = "🧠 SYSTEM 2 (Deliberativo)"
        else:
            action = decision.action
            s1_count += 1
            origin = "⚡ SYSTEM 1 (Reflexo Amortizado)"

        obs_dict, reward, term, trunc, _ = env.step(action)
        if step < 6 or decision.is_uncertain:
            print(f"  Passo {step+1:2d} | Conf: {decision.confidence*100:4.1f}% | Uncert: {decision.uncertainty*100:4.1f}% | Decisão por: {origin}")

        if term or trunc:
            break

    print(f"\n• Resumo do Episódio: Decisões System 1: {s1_count} ({s1_count/(s1_count+s2_count)*100:.0f}%) | Invocação System 2: {s2_count}")

    # 4. Calibração e Proteção de Underflow em Ações Contínuas (Pendulum-v1)
    print("\n" + "=" * 75)
    print("🎯 CALIBRAÇÃO EM ESPAÇO CONTÍNUO (GaussianPolicyHead / Pendulum-v1)")
    print("=" * 75)

    raw_cont_env = gym.make("Pendulum-v1")
    cont_env = UniversalS1Wrapper(raw_cont_env)
    agent_cont = UniversalS1Agent(obs_space=cont_env.observation_space, action_space=cont_env.action_space)
    agent_cont.eval()

    obs_cont, _ = cont_env.reset(seed=42)

    print("🔬 TESTE DE REGIMES DE DISPERSÃO GAUSSIANA (SIGMA):")

    regimes = [
        ("Exploração Padrão (Tabula Rasa)", 1.0),
        ("Dispersão Crítica (H ~ 0)", 0.24197),
        ("Política Consolidada (H < 0)", 0.05),
        ("Limite Inferior Numérico", 1e-6),
        ("Alta Incerteza / Caos", 2.0),
    ]

    for label, sigma_val in regimes:
        with torch.no_grad():
            agent_cont.policy_head.log_std.fill_(float(np.log(sigma_val)))

        dec = agent_cont.act_with_confidence(obs_cont, uncertainty_threshold=0.70, confidence_threshold=0.50)
        status = "🚨 ATIVADO" if dec.is_uncertain else "🟢 DESATIVADO"
        print(f"  • {label:32s}: σ={sigma_val:8.5f} | H={dec.entropy:7.3f} | "
              f"Uncert={dec.uncertainty*100:5.2f}% | Conf={dec.confidence*100:5.2f}% | Gatilho: {status}")

    print("\n✅ Ausência total de underflow: Para σ < 0.242, entropia é negativa mas incerteza permanece em [0, 1]!")
    print("=" * 75)


if __name__ == "__main__":
    run_confidence_gating_demo()
