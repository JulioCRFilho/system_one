#!/usr/bin/env bash
# Atalho rápido para treinar o Cubo Mágico 3x3 no System 1 Engine
# Uso:
#   ./scripts/train_rubiks.sh               # Treina macro-ações (15000 passos)
#   ./scripts/train_rubiks.sh atomic        # Treina giros atômicos
#   ./scripts/train_rubiks.sh macro 30000 3 # Treina macro com 30k passos e scramble depth 3

set -e

cd "$(dirname "$0")/.."

MODE="${1:-macro}"
STEPS="${2:-15000}"
DEPTH="${3:-2}"

echo "================================================================="
echo "🎲 INICIANDO TREINAMENTO DO CUBO MÁGICO (SYSTEM 1 ENGINE)"
echo "   Modo: $MODE | Passos: $STEPS | Scramble Depth: $DEPTH"
echo "================================================================="

.venv/bin/python examples/13_train_rubiks_cube.py --mode "$MODE" --steps "$STEPS" --scramble-depth "$DEPTH"
