#!/bin/bash
# ==============================================================================
# System 1 HUD - Launcher Script
# ==============================================================================

PROJECT_DIR="/Users/juliocesarreisfilho/Projects/system_one"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
LOG_FILE="/tmp/system1_hud.log"
PORT=8050
URL="http://127.0.0.1:$PORT"

# 1. Verifica se a porta 8050 já está em escuta
if /usr/sbin/lsof -i :$PORT -sTCP:LISTEN >/dev/null 2>&1 || nc -z 127.0.0.1 $PORT >/dev/null 2>&1; then
    echo "[i] O servidor HUD já está ativo na porta $PORT."
else
    echo "[+] Iniciando o servidor HUD em segundo plano..."
    cd "$PROJECT_DIR" || exit 1
    nohup "$PYTHON_BIN" -m system1_engine.hud > "$LOG_FILE" 2>&1 &
    
    # Aguarda a porta 8050 responder (até 6 segundos)
    for i in {1..60}; do
        if nc -z 127.0.0.1 $PORT >/dev/null 2>&1; then
            break
        fi
        sleep 0.1
    done
fi

# 2. Abre a interface no navegador padrão do macOS
/usr/bin/open "$URL"
