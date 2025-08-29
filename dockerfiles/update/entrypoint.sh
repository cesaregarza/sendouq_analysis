#!/usr/bin/env bash
set -euo pipefail

# Defaults
WEEKS_BACK=${WEEKS_BACK:-2}
SINCE_DAYS=${SINCE_DAYS:-120}
ONLY_RANKED=${ONLY_RANKED:-true}
SAVE_TO_DB=${SAVE_TO_DB:-true}
WRITE_PARQUET=${WRITE_PARQUET:-false}
BUILD_VERSION=${BUILD_VERSION:-}

DATA_DIR=${DATA_DIR:-/app/data/tournaments}
COMPILED_OUT=${COMPILED_OUT:-/app/compiled}

echo "[entrypoint] rankings_update"
echo "  weeks_back    = ${WEEKS_BACK}"
echo "  since_days    = ${SINCE_DAYS}"
echo "  only_ranked   = ${ONLY_RANKED}"
echo "  save_to_db    = ${SAVE_TO_DB}"
echo "  write_parquet = ${WRITE_PARQUET}"
echo "  data_dir      = ${DATA_DIR}"
echo "  compiled_out  = ${COMPILED_OUT}"
echo "  db_url (short)= ${RANKINGS_DATABASE_URL:+set}/${DATABASE_URL:+set}"
echo "  schema        = ${RANKINGS_DB_SCHEMA:-comp_rankings}"

ARGS=(
  --weeks-back "${WEEKS_BACK}"
  --data-dir "${DATA_DIR}"
  --compiled-out "${COMPILED_OUT}"
  --since-days "${SINCE_DAYS}"
)

if [[ "${ONLY_RANKED}" == "true" ]]; then
  ARGS+=(--only-ranked)
fi

if [[ "${SAVE_TO_DB}" == "true" ]]; then
  ARGS+=(--save-to-db)
fi

if [[ -n "${BUILD_VERSION}" ]]; then
  ARGS+=(--build-version "${BUILD_VERSION}")
fi

if [[ "${WRITE_PARQUET}" == "true" ]]; then
  ARGS+=(--write-parquet)
fi

# Prefer module execution to avoid reliance on console_scripts install
exec poetry run python -m rankings.cli.update "${ARGS[@]}" "$@"
