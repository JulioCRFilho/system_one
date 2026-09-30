#!/usr/bin/env python3
"""Task 7: Execução Sequencial de Todas as Tasks do System 1.

Roda a suíte completa de demonstração operacional:
1. Benchmark de Latência CPU (<= 0.8 ms)
2. Avaliação do Checkpoint Treinado (CartPole-v1)
3. Transferência de Conhecimento com Tronco Congelado (Acrobot-v1)
4. Controle Contínuo com GaussianPolicyHead (Pendulum-v1)
5. Pipeline de Percepção Visual IMPALA (Entrada 2*C, 84, 84)
"""

from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_BIN = sys.executable

TASKS = [
    ("Task 1: Benchmark de Latência CPU", "examples/01_benchmark_latency.py"),
    ("Task 2: Avaliação de Checkpoint Treinado", "examples/02_evaluate_cartpole.py"),
    ("Task 4: Transferência com Tronco Congelado", "examples/04_transfer_learning_acrobot.py"),
    ("Task 5: Controle Contínuo (Pendulum-v1)", "examples/05_continuous_action_pendulum.py"),
    ("Task 6: Percepção Visual (IMPALA)", "examples/06_visual_observation_impala.py"),
    ("Task 8: Adaptadores 3 Níveis (Window/Mem/Native)", "examples/08_adapters_three_levels.py"),
]


def run_all() -> None:
    print("=" * 80)
    print("🌟 UNIVERSAL SYSTEM 1 ENGINE - DEMONSTRAÇÃO GERAL DE TASKS OPERACIONAIS")
    print("=" * 80)

    results = []
    t_global_start = time.time()

    for name, script in TASKS:
        script_path = PROJECT_ROOT / script
        print(f"\n▶ Executando: {name} ({script})...")
        t0 = time.time()
        res = subprocess.run(
            [PYTHON_BIN, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0

        if res.returncode == 0:
            print(f"✔ Sucesso em {elapsed:.2f}s")
            results.append((name, "PASS", elapsed))
        else:
            print(f"✖ Falha em {elapsed:.2f}s:")
            print(res.stderr)
            results.append((name, "FAIL", elapsed))

    t_global_total = time.time() - t_global_start

    print("\n" + "=" * 80)
    print(f"🏁 PAINEL GERAL DE EXECUÇÃO ({t_global_total:.2f}s totais)")
    print("=" * 80)
    for name, status, duration in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"{icon} {name:<45} [{status}] ({duration:.2f}s)")
    print("=" * 80)


if __name__ == "__main__":
    run_all()
