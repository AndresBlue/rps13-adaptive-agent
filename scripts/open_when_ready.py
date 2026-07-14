"""Wait until a TCP port accepts connections, then open a URL in the browser."""

from __future__ import annotations

import argparse
import socket
import sys
import time
import webbrowser

from tqdm import tqdm


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    deadline = time.time() + args.timeout
    steps = max(1, int(args.timeout * 2))
    with tqdm(total=steps, desc="Esperando servidor", unit="check", leave=False) as bar:
        while time.time() < deadline:
            if port_open(args.host, args.port):
                bar.close()
                webbrowser.open(args.url)
                return 0
            time.sleep(0.5)
            bar.update(1)

    print(f"No se pudo abrir el navegador: {args.url} no respondio a tiempo.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
