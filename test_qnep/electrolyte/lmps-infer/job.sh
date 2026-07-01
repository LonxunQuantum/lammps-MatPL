#!/bin/sh
#SBATCH -p 3080ti,new3080ti,3090,q3,q4
#SBATCH -J qnep-infer
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1

set -eu

echo "SLURM_SUBMIT_DIR is $SLURM_SUBMIT_DIR"
echo "Running on nodes: $SLURM_NODELIST"

module purge
module load lmps/qnep-86
which lmp

export OMP_NUM_THREADS=1

mkdir -p logs outputs/ewald outputs/pppm

for input in inputs/*.lmp; do
  case_name=$(basename "$input" .lmp)
  echo "===== RUN $case_name ====="
  mpirun -np 1 lmp -k on g 1 -sf kk -pk kokkos -log "logs/${case_name}.log" -in "$input" > "logs/${case_name}.out" 2>&1
  echo "===== DONE $case_name ====="
done

echo "Job $SLURM_JOB_ID done at $(date)"
