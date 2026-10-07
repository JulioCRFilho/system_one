#!/usr/bin/env bash
# Atalho rápido para treinar o FrozenLake-v1 com Curriculum Learning Reverso
# Uso:
#   ./scripts/train_frozenlake.sh          # Treina com 15000 passos e curriculum
#   ./scripts/train_frozenlake.sh 25000    # Treina com 25000 passos

set -e

cd "$(dirname "$0")/.."

STEPS="${1:-15000}"
EXTRA_ARGS="${@:2}"

echo "================================================================="
echo "❄️ INICIANDO TREINAMENTO DO FROZENLAKE (CURRICULUM LEARNING)"
echo "   Passos: $STEPS | Meta: Níveis 1 a 4 (Spawn Reverso)"
echo "================================================================="

.venv/bin/python examples/14_train_frozenlake.py --steps "$STEPS" --curriculum $EXTRA_ARGS
