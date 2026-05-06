#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCLAW_REPO="${OPENCLAW_REPO:-$PWD/openclaw}"

if [[ ! -d "$OPENCLAW_REPO" ]]; then
  echo "OPENCLAW_REPO does not exist: $OPENCLAW_REPO" >&2
  echo "Clone OpenClaw first, for example:" >&2
  echo "  git clone https://github.com/openclaw/openclaw openclaw" >&2
  exit 2
fi

mkdir -p "$OPENCLAW_REPO/benchmarks"
rm -rf "$OPENCLAW_REPO/benchmarks/memory_mvp"
cp -R "$SCRIPT_DIR/benchmarks/memory_mvp" "$OPENCLAW_REPO/benchmarks/memory_mvp"

PATCHED_EMBEDDING_OPS="$SCRIPT_DIR/patches/extensions/memory-core/src/memory/manager-embedding-ops.ts"
TARGET_EMBEDDING_OPS="$OPENCLAW_REPO/extensions/memory-core/src/memory/manager-embedding-ops.ts"
if [[ -f "$PATCHED_EMBEDDING_OPS" ]]; then
  cp "$PATCHED_EMBEDDING_OPS" "$TARGET_EMBEDDING_OPS"
fi

echo "Installed OpenClaw memory benchmark into $OPENCLAW_REPO/benchmarks/memory_mvp"
