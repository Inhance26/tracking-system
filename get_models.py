#!/usr/bin/env python3
"""Download YOLOX ONNX models for the --detector yolox backend.

    python get_models.py            # the recommended one (yolox_s)
    python get_models.py tiny       # smaller and faster, misses more
    python get_models.py m          # bigger and slower, finds a little more
    python get_models.py all

These come from the official YOLOX GitHub release. They run through OpenCV's
DNN module, so no PyTorch and no Ultralytics is needed - which is the whole
point of this backend.
"""

from __future__ import annotations

import os
import sys
import urllib.request

BASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
MODELS = {
    "tiny": ("yolox_tiny.onnx", 20),   # name, approx MB
    "s":    ("yolox_s.onnx", 36),
    "m":    ("yolox_m.onnx", 97),
}
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "models")


def _progress(done: int, block: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100, done * block * 100 // total)
    sys.stdout.write(f"\r    {pct:3d}%")
    sys.stdout.flush()


def fetch(key: str) -> bool:
    name, mb = MODELS[key]
    path = os.path.join(DEST, name)
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        print(f"  {name} already here")
        return True
    os.makedirs(DEST, exist_ok=True)
    print(f"  downloading {name} (~{mb} MB)")
    try:
        urllib.request.urlretrieve(f"{BASE}/{name}", path, _progress)
        print(f"\r    done -> models/{name}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\r    failed: {exc}")
        if os.path.exists(path):
            os.remove(path)
        return False


def main() -> int:
    args = [a.lower() for a in sys.argv[1:]] or ["s"]
    keys = list(MODELS) if "all" in args else [a for a in args if a in MODELS]
    if not keys:
        print(f"Unknown model. Choose from: {', '.join(MODELS)} (or 'all')")
        return 1
    print()
    ok = all(fetch(k) for k in keys)
    print("\n  Then run:  python app.py --detector yolox\n" if ok else
          "\n  Some downloads failed - check your internet connection.\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
