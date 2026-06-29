#!/bin/sh
#SBATCH -p q4
#SBATCH -J heat_flux
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -o out

echo "Starting job $SLURM_JOB_ID at " `date`
echo "SLURM_SUBMIT_DIR is $SLURM_SUBMIT_DIR"
echo "Running on nodes: $SLURM_NODELIST"

module purge
module load openmpi/4.1.6 cuda/11.6-share
source /opt/rh/devtoolset-8/enable
export PATH=$PATH:/share/public/PWMLFF_test_data/pfsuo/MatPL-pro/AMPERE86/bin
export NEP_GPU_LIB_PATH=/share/public/PWMLFF_test_data/pfsuo/MatPL-pro/AMPERE86/lib64/libnep_gpu.so
export NEP_ANN_EXEC_KIND=tc
export OMP_NUM_THREADS=1

HEATFLUX_MODE=${HEATFLUX_MODE:-1}
THERMO_FLUX_MODE=${THERMO_FLUX_MODE:-1}

echo "heatflux_mode=$HEATFLUX_MODE"
echo "thermo_flux_mode=$THERMO_FLUX_MODE"

mpirun -np $SLURM_NPROCS --bind-to numa --map-by ppr:$SLURM_NPROCS:node \
	lmp_mpi -k on g $SLURM_NPROCS -sf kk -pk kokkos \
	-var heatflux_mode $HEATFLUX_MODE \
	-var thermo_flux_mode $THERMO_FLUX_MODE \
	-in graphene.in > lmp.out 2>&1

echo "Job $SLURM_JOB_ID done at " `date`
