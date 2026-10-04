import argparse
import sys
import time

from system1_engine.hud.server import HUDServer


def main() -> None:
    parser = argparse.ArgumentParser(description="System 1 Engine — Visual Operational HUD")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Endereço de rede (padrão: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8050, help="Porta HTTP do HUD (padrão: 8050)")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=False,
        help="Não abre o navegador padrão automaticamente",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token de autenticação para acesso remoto seguro (ou defina S1_AUTH_TOKEN)",
    )

    args = parser.parse_args()

    should_open_browser = not args.no_browser and sys.stdin.isatty()
    server = HUDServer(
        host=args.host,
        port=args.port,
        open_browser=should_open_browser,
        auth_token=args.token,
    )
    server.start()

    print("⌨️  Pressione Ctrl+C no terminal a qualquer momento para encerrar o HUD...")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nEncerrando servidor HUD...")
    finally:
        server.stop()
        print("Servidor HUD encerrado com sucesso.")


if __name__ == "__main__":
    main()
