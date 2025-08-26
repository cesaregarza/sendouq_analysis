#!/usr/bin/env bash
set -euo pipefail

ISO_YEAR=$(date +%G)
ISO_WEEK=$(date +%V)
DEFAULT_OUT="data/tournaments/ranked_${ISO_YEAR}_${ISO_WEEK}"
OUT_DIR=${RANKED_OUTPUT_DIR:-$DEFAULT_OUT}

echo "[entrypoint] Ranked scrape for ISO ${ISO_YEAR}/${ISO_WEEK} (past two weeks)"
echo "[entrypoint] Output dir: ${OUT_DIR}"

if [[ -z "${SENDOU_KEY:-}" ]]; then
  echo "[entrypoint] Warning: SENDOU_KEY not set; ensure .env or env var is provided." >&2
fi

exec poetry run ranked \
  --weeks-back 2 \
  --skip-existing \
  --output-dir "${OUT_DIR}" \
  "$@"

