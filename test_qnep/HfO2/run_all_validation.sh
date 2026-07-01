#!/bin/bash
# Run the HfO2 MatPL/GPUMD validation workflow.

set -u

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

RESULT_DIR="$ROOT_DIR/validation_results"
mkdir -p "$RESULT_DIR"

LOG_FILE="$RESULT_DIR/run_all_validation.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "Validation root: $ROOT_DIR"
echo "Started at: $(date)"

OVERALL_STATUS=0

run_step() {
  local name="$1"
  shift

  echo
  echo "===== ${name} ====="
  "$@"
  local rc=$?
  echo "===== ${name} exit_code=${rc} ====="
  if [ "$rc" -ne 0 ]; then
    OVERALL_STATUS=1
  fi
  return 0
}

run_in_dir() {
  local dir="$1"
  shift
  (
    cd "$dir"
    "$@"
  )
}

run_step "step1 MatPL nep4/nep5 ewald/pppm inference" \
  run_in_dir "$ROOT_DIR/MatPL-infer" sbatch --wait run.sh

run_step "step2 MatPL generated results vs ref-result, threshold=1e-5" \
  run_in_dir "$ROOT_DIR/MatPL-infer" ./compare_matpl_ref.py --threshold 1e-5

run_step "step3 GPUMD vs MatPL nep4 ewald force/bec, threshold=1e-3" \
  run_in_dir "$ROOT_DIR/MatPL-infer" ./compare_gpumd_matpl.py \
    --matpl-dir test_result_nep4_ewald \
    --gpumd-dir ../GPUMD \
    --threshold 1e-3 \
    --output ../validation_results/gpumd_matpl_summary.csv \
    --json-output ../validation_results/gpumd_matpl_summary.json

cat > "$RESULT_DIR/validation_status.txt" <<EOF
overall_status=$OVERALL_STATUS
finished_at=$(date)
log_file=$LOG_FILE
matpl_ref_summary=$RESULT_DIR/matpl_ref_summary.csv
gpumd_matpl_summary=$RESULT_DIR/gpumd_matpl_summary.csv
EOF

echo
echo "Finished at: $(date)"
echo "Overall status: $OVERALL_STATUS"
echo "Summary files:"
cat "$RESULT_DIR/validation_status.txt"

exit "$OVERALL_STATUS"
