#!/usr/bin/env bash
# Atalho rápido para treinar o Cubo Mágico 3x3 no System 1 Engine
# Uso:
#   ./scripts/train_rubiks.sh               # Treina macro-ações (15000 passos)
#   ./scripts/train_rubiks.sh atomic        # Treina giros atômicos
#   ./scripts/train_rubiks.sh macro 30000 3 # Treina macro com 30k passos e scramble depth 3

set -e

cd "$(dirname "$0")/.."

MODE="${1:-atomic}"
STEPS="${2:-40000}"
DEPTH="${3:-4}"
EXTRA_ARGS="${@:4}"

echo "================================================================="
echo "🎲 INICIANDO TREINAMENTO DO CUBO MÁGICO (SYSTEM 1 ENGINE)"
echo "   Modo: $MODE | Passos: $STEPS | Max Depth: $DEPTH (Curriculum)"
echo "================================================================="

.venv/bin/python examples/13_train_rubiks_cube.py --mode "$MODE" --steps "$STEPS" --max-depth "$DEPTH" --curriculum $EXTRA_ARGS
