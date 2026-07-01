#!/bin/sh
#SBATCH -p 3080ti,new3080ti,3090,q3,q4
#SBATCH -J qnep4-compare
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1

set -eu

echo "SLURM_SUBMIT_DIR is $SLURM_SUBMIT_DIR"
echo "Running on nodes: $SLURM_NODELIST"
echo "Starting at $(date)"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
  cd "$SCRIPT_DIR"
fi

if [ ! -f run_lmps_compare.py ]; then
  echo "Cannot find run_lmps_compare.py in $(pwd)" >&2
  echo "Please submit this job from the lmps-infer-nep4 directory." >&2
  exit 2
fi

set +u
source ../env.sh
set -u

which python
which lmp

export OMP_NUM_THREADS=1

python run_lmps_compare.py "$@"

echo "Finished at $(date)"
