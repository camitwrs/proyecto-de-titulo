#!/bin/bash
# Smoke test de EfficientNet-B4 en el HPC Oceano PUCV (cola de pruebas, 1 GPU).
# Uso, desde ~/proyecto_de_titulo en el nodo de login:
#   mkdir -p logs && sbatch jobs/smoke_test_hpc.sbatch
#SBATCH -J smoke_b4_hpc
#SBATCH -p GPU
#SBATCH --qos=test
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%j_%x.out
#SBATCH -e logs/%j_%x.err

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tesis

# Sin esto TensorFlow no encuentra las librerias CUDA instaladas por pip.
NV_DIR=$(python -c "import nvidia; print(list(nvidia.__path__)[0])")
export LD_LIBRARY_PATH=$(ls -d "$NV_DIR"/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}

# Slurm no administra las GPU (Gres=null): las dos son visibles, se fija una por trabajo.
export CUDA_VISIBLE_DEVICES=0

cd "$HOME/proyecto_de_titulo"

echo "nodo: $(hostname)"
echo "fecha: $(date -Is)"
nvidia-smi

python scripts/smoke_test_b4.py

cp reports/smoke_test_b4_report.json reports/smoke_test_b4_report_hpc.json
echo "listo: reports/smoke_test_b4_report_hpc.json"
