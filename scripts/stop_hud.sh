#!/usr/bin/env bash
if pkill -f "python.*system1_engine.hud" >/dev/null 2>&1; then
    echo "[✓] Servidor System 1 HUD encerrado com sucesso."
else
    echo "[i] Nenhum servidor System 1 HUD ativo em execução."
fi
