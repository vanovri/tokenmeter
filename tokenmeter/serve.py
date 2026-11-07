"""Run the proxy on 127.0.0.1 plus optional extra host IPs (e.g. the WSL vEthernet IP),
without exposing it on 0.0.0.0. Usage: python -m tokenmeter.serve --port 8787 --extra-host 172.31.112.1"""
from __future__ import annotations

import argparse
import socket
from collections.abc import Sequence

import uvicorn

from .proxy import create_app


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TokenMeter multi-socket proxy launcher")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--extra-host", action="append", default=[])
    return ap.parse_args(argv)


def resolve_hosts(host: str, extra_hosts: Sequence[str]) -> list[str]:
    """Deduplicate primary and extra bind hosts while preserving order."""
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in [host, *extra_hosts]:
        cleaned = candidate.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return ordered


def bind_sockets(hosts: Sequence[str], port: int) -> list[socket.socket]:
    if isinstance(port, bool) or not (0 <= port <= 65535):
        raise ValueError(f"port must be between 0 and 65535, got {port!r}")
    socks: list[socket.socket] = []
    try:
        for h in hosts:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind((h, port))
                s.listen(128)
                s.set_inheritable(True)
            except OSError:
                s.close()
                raise
            bound_port = s.getsockname()[1]
            socks.append(s)
            print(f"TokenMeter listening on http://{h}:{bound_port}/v1", flush=True)
    except OSError:
        for opened in socks:
            opened.close()
        raise
    return socks


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    hosts = resolve_hosts(args.host, args.extra_host)
    socks = bind_sockets(hosts, args.port)
    try:
        server = uvicorn.Server(uvicorn.Config(create_app(), log_level="info"))
        server.run(sockets=socks)
    finally:
        for s in socks:
            s.close()


if __name__ == "__main__":
    main()
