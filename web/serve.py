#!/usr/bin/env python3
"""Serve the browser classifier locally for testing.

    python3 web/serve.py            # then open http://127.0.0.1:8765/

Any static web server can host the web/ folder. This one only adds the two
headers that let the browser use several CPU threads, and binds to this
computer only.
"""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".js": "text/javascript",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
    }

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, *args) -> None:  # keep the terminal quiet
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    handler = partial(Handler, directory=str(args.root.resolve()))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError as error:
        raise SystemExit(
            f"Port {args.port} is not available ({error.strerror}). Another copy may already be "
            f"running: try http://127.0.0.1:{args.port}/ or choose another port with --port."
        )
    with server:
        print(f"Serving {args.root.resolve()} at http://127.0.0.1:{args.port}/  (Ctrl+C to stop)")
        server.serve_forever()


if __name__ == "__main__":
    main()
