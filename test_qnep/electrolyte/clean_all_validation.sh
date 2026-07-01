#!/bin/bash
# Clean generated electrolyte validation artifacts.

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

if [ "${1:-}" != "--yes" ] || [ "$#" -ne 1 ]; then
  echo "Usage: ./clean_all_validation.sh --yes" >&2
  echo "This removes validation reports, generated MatPL results, LAMMPS auto outputs, and local slurm logs." >&2
  exit 2
fi

echo "Cleaning validation artifacts under $ROOT_DIR"

rm -rf validation_results

rm -rf \
  MatPL-infer/test_result_nep4_ewald \
  MatPL-infer/test_result_nep4_pppm \
  MatPL-infer/test_result_nep5_ewald \
  MatPL-infer/test_result_nep5_pppm \
  MatPL-infer/__pycache__

rm -f \
  MatPL-infer/compare_gpumd_matpl_summary.csv \
  MatPL-infer/compare_gpumd_matpl_summary.json \
  MatPL-infer/slurm-*.out \
  lmps-infer/slurm-*.out \
  lmps-infer-nep4/slurm-*.out

if [ -x lmps-infer/clean_lmps_compare.sh ]; then
  (cd lmps-infer && ./clean_lmps_compare.sh --yes)
fi

if [ -x lmps-infer-nep4/clean_lmps_compare.sh ]; then
  (cd lmps-infer-nep4 && ./clean_lmps_compare.sh --yes)
fi

echo "clean done"
