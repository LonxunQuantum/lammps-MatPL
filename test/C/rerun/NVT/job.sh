#!/bin/sh
#SBATCH -p cpu
#SBATCH -J baseline
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o out

echo "Starting job $SLURM_JOB_ID at " `date`
echo "SLURM_SUBMIT_DIR is $SLURM_SUBMIT_DIR"
echo "Running on nodes: $SLURM_NODELIST"

module load lammps/NEP_CPU
export OMP_NUM_THREADS=1

mpirun -np $SLURM_NPROCS lmp -in lmp.in > lmp.out 2>&1

echo "Job $SLURM_JOB_ID done at " `date`
