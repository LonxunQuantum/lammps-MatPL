#!/bin/sh
# Clean GPUMD-vs-MatPL comparison reports in this directory.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

if [ "${1:-}" != "--yes" ] || [ "$#" -ne 1 ]; then
  echo "Usage: ./clean_gpumd_matpl_compare.sh --yes" >&2
  echo "This removes: compare_gpumd_matpl_summary.csv compare_gpumd_matpl_summary.json" >&2
  exit 2
fi

rm -f compare_gpumd_matpl_summary.csv compare_gpumd_matpl_summary.json
echo "clean done"
