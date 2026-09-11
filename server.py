"""Standard-library HTTP server: MJPEG video, stats JSON, zone CRUD, static UI.

No FastAPI/Flask on purpose - the whole app installs with nothing but
opencv-python (plus ultralytics if you use the YOLO backend).
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from zones import save_zones, zones_from_payload

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
BOUNDARY = "retailframe"


def _jsonable(obj):
    """Last-resort coercion for anything json doesn't know (numpy scalars)."""
    item = getattr(obj, "item", None)
    if callable(item):
        return item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def make_handler(state, pipeline, cfg):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "RetailTracker/1.0"

        def log_message(self, fmt, *args):  # quieter console
            if cfg.verbose:
                super().log_message(fmt, *args)

        # -- helpers ----------------------------------------------------------
        def _send(self, code, body: bytes, ctype: str, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, code=200):
            # default=_jsonable: a single numpy scalar leaking in from a detector
            # used to raise here, and a raised exception means no response at all -
            # the dashboard's fetch() fails and it silently keeps showing stale
            # numbers. Coerce instead of dying.
            body = json.dumps(payload, default=_jsonable).encode("utf-8")
            self._send(code, body, "application/json")

        def _static(self, filename, ctype):
            path = os.path.join(WEB_DIR, filename)
            if not os.path.exists(path):
                self._send(404, b"not found", "text/plain")
                return
            with open(path, "rb") as fh:
                self._send(200, fh.read(), ctype)

        # -- routes -----------------------------------------------------------
        def do_GET(self):
            route = urlparse(self.path).path.rstrip("/") or "/"

            if route == "/":
                return self._static("dashboard.html", "text/html; charset=utf-8")
            if route == "/editor":
                return self._static("editor.html", "text/html; charset=utf-8")
            if route == "/api/stats":
                return self._json(state.read_stats())
            if route == "/api/zones":
                return self._json({
                    "zones": [
                        {"id": z.id, "name": z.name, "points": z.points, "color": z.color}
                        for z in pipeline.current_zones()
                    ]
                })
            if route == "/snapshot.jpg":
                jpg = state.read_raw_jpeg()
                if jpg is None:
                    return self._send(503, b"no frame yet", "text/plain")
                return self._send(200, jpg, "image/jpeg")
            if route == "/video":
                return self._mjpeg()
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            route = urlparse(self.path).path.rstrip("/") or "/"
            if route != "/api/zones":
                return self._send(404, b"not found", "text/plain")
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                zones = zones_from_payload(payload)
            except Exception as exc:  # noqa: BLE001 - report any parse problem to the UI
                return self._json({"ok": False, "error": str(exc)}, 400)
            save_zones(cfg.zones_path, zones)
            pipeline.reload_zones()
            return self._json({"ok": True, "count": len(zones)})

        # -- MJPEG ------------------------------------------------------------
        def _mjpeg(self):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type",
                             f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.end_headers()
            last = None
            try:
                while True:
                    jpg = state.read_jpeg()
                    if jpg is None or jpg is last:
                        time.sleep(0.01)
                        continue
                    last = jpg
                    self.wfile.write(
                        f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                    )
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser closed the tab

    return Handler


def serve(state, pipeline, cfg):
    handler = make_handler(state, pipeline, cfg)
    httpd = ThreadingHTTPServer((cfg.host, cfg.port), handler)
    httpd.daemon_threads = True
    return httpd
