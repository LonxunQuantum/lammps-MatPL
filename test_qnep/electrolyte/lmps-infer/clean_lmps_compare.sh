#!/bin/sh
# Clean generated artifacts for this LAMMPS compare directory.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

if [ "${1:-}" != "--yes" ] || [ "$#" -ne 1 ]; then
  echo "Usage: ./clean_lmps_compare.sh --yes" >&2
  echo "This removes: auto_inputs auto_outputs auto_logs compare_results __pycache__ slurm-*.out" >&2
  exit 2
fi

for target in auto_inputs auto_outputs auto_logs compare_results __pycache__; do
  if [ -e "$target" ]; then
    echo "removing: $target"
    rm -rf -- "$target"
  fi
done

for target in slurm-*.out; do
  [ -e "$target" ] || continue
  echo "removing: $target"
  rm -f -- "$target"
done

echo "clean done"
