#!/usr/bin/env bash
set -euo pipefail

ISO_YEAR=$(date +%G)
ISO_WEEK=$(date +%V)
DEFAULT_OUT="data/tournaments/weekly_${ISO_YEAR}_${ISO_WEEK}"
OUT_DIR=${WEEKLY_OUTPUT_DIR:-$DEFAULT_OUT}

echo "[entrypoint] Weekly scrape for ISO ${ISO_YEAR}/${ISO_WEEK}"
echo "[entrypoint] Output dir: ${OUT_DIR}"

if [[ -z "${SENDOU_KEY:-}" ]]; then
  echo "[entrypoint] Warning: SENDOU_KEY not set in environment; ensure .env is mounted or variable is provided." >&2
fi

exec poetry run scrape_weekly \
  --skip-existing \
  --output-dir "${OUT_DIR}" \
  "$@"

