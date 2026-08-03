#!/usr/bin/env python3
"""Verify every file listed in MANIFEST.sha256."""

from __future__ import annotations

import hashlib
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "MANIFEST.sha256"
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative_path = line.split("  ", 1)
        path = root / relative_path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(f"Checksum mismatch: {relative_path}")
        checked += 1
    print(f"Verified {checked} artifact files.")


if __name__ == "__main__":
    main()
