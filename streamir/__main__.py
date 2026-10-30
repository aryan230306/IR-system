"""CLI:  python -m streamir {serve|generate|bench|search}"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="streamir", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sv = sub.add_parser("serve", help="run the API + dashboard")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--source", default=None, help="synthetic | jsonl:<path> | rss:<url,url> | none")
    sv.add_argument("--rate", type=float, default=None, help="synthetic docs/second")
    sv.add_argument("--window", type=float, default=None, help="retention window (s)")
    sv.add_argument("--no-wal", action="store_true")

    ge = sub.add_parser("generate", help="write a synthetic JSONL corpus")
    ge.add_argument("--n", type=int, default=5000)
    ge.add_argument("--out", default="data/sample.jsonl")
    ge.add_argument("--seed", type=int, default=7)
    ge.add_argument("--rate", type=float, default=50.0)

    be = sub.add_parser("bench", help="run the benchmark / evaluation suite")
    be.add_argument("--n", type=int, default=20000)

    a, rest = ap.parse_known_args(argv)
    if a.cmd == "serve":
        import uvicorn
        from .api import create_app
        from .config import Settings
        s = Settings()
        if a.source: s.source = a.source
        if a.rate: s.rate = a.rate
        if a.window: s.window_seconds = a.window
        if a.no_wal: s.wal_path = None
        uvicorn.run(create_app(s), host=a.host, port=a.port, log_level="info")
    elif a.cmd == "generate":
        from .sources import SyntheticNewsSource
        import os
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            for d in SyntheticNewsSource(rate=a.rate, seed=a.seed, n_docs=a.n, realtime=False).generate():
                fh.write(json.dumps(d.to_dict()) + "\n")
        print(f"wrote {a.n} docs -> {a.out}")
    elif a.cmd == "bench":
        sys.path.insert(0, ".")
        from bench.benchmark import run
        run(a.n)


if __name__ == "__main__":
    main()
