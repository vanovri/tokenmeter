"""Run the proxy on 127.0.0.1 plus optional extra host IPs (e.g. the WSL vEthernet IP),
without exposing it on 0.0.0.0.  Usage: python -m tokenmeter.serve --port 8787 --extra-host 172.31.112.1"""
import argparse, socket, uvicorn
from .proxy import create_app

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--extra-host", action="append", default=[])
    a = ap.parse_args(argv)
    socks = []
    for h in [a.host] + [x for x in a.extra_host if x and x != a.host]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((h, a.port)); s.listen(128); s.set_inheritable(True)
        socks.append(s); print(f"TokenMeter listening on http://{h}:{a.port}/v1", flush=True)
    server = uvicorn.Server(uvicorn.Config(create_app(), log_level="info"))
    server.run(sockets=socks)

if __name__ == "__main__":
    main()
