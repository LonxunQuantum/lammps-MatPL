#!/bin/bash
#SBATCH --job-name=qnep12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=3090,q4,4090,4090t

set -eu

echo "SLURM_SUBMIT_DIR is ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Starting job ${SLURM_JOB_ID:-manual} at " `date`
echo "Running on nodes: ${SLURM_NODELIST:-local}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

start=$(date +%s)
set +u
source ../env.sh
set -u

if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
else
    MASTER_ADDR=$(hostname)
fi
#MASTER_PORT=12355

# 动态分配空闲端口
function get_free_port() {
    python -c 'import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}
MASTER_PORT=$(get_free_port)

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

echo "addrs: $MASTER_ADDR"
echo "port:  $MASTER_PORT"
echo "tasks: ${SLURM_NTASKS:-1}"

write_test_json() {
    local model_file="$1"
    local kspace_mode="$2"
    local result_dir="$3"

    python - "$model_file" "$kspace_mode" "$result_dir" <<'PY'
import json
import sys
from pathlib import Path

path = Path("test.json")
if path.exists():
    data = json.loads(path.read_text())
else:
    data = {}

data.update(
    {
        "model_type": "NEP",
        "atom_type": [
            1, 3, 5, 6, 7, 8, 9, 11, 12, 14,
            15, 16, 17, 19, 20, 26, 30, 35, 53,
        ],
        "model_load_file": sys.argv[1],
        "format": "extxyz",
        "kspace": sys.argv[2],
        "test_data": ["../configs/train.xyz"],
        "test_dir_name": sys.argv[3],
    }
)

path.write_text(json.dumps(data, indent=4) + "\n")
PY
}

for model in nep4 nep5; do
    for kspace in ewald pppm; do
        model_file="../${model}.txt"
        result_dir="test_result_${model}_${kspace}"

        echo "===== RUN ${model} ${kspace} -> ${result_dir} ====="
        write_test_json "$model_file" "$kspace" "$result_dir"
        cat test.json
        MATPL test test.json
        echo "===== DONE ${model} ${kspace} ====="
    done
done


echo "Job ${SLURM_JOB_ID:-manual} done at " `date`

end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds
exit 0
