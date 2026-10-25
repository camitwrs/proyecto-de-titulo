#!/bin/bash
# Piloto tecnico de M1, Fase 1 (backbone congelado), solo train y validation.
# Uso, desde ~/proyecto_de_titulo en el nodo de login:
#   mkdir -p logs && sbatch jobs/train_m1_fase1.sbatch
# Otra semilla o ruta de datos:
#   sbatch --export=ALL,SEED=123,DATA_ROOT=/work/usuario/isic jobs/train_m1_fase1.sbatch
#SBATCH -J m1_fase1
#SBATCH -p GPU
#SBATCH --qos=normal-gpu
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH -o logs/%j_%x.out
#SBATCH -e logs/%j_%x.err

set -eo pipefail

SEED="${SEED:-42}"
DATA_ROOT="${DATA_ROOT:-/work/$USER/isic}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tesis

# Sin esto TensorFlow no encuentra las librerias CUDA instaladas por pip.
NV_DIR=$(python -c "import nvidia; print(list(nvidia.__path__)[0])")
export LD_LIBRARY_PATH=$(ls -d "$NV_DIR"/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}

# Slurm no administra las GPU (Gres=null): las dos son visibles, se fija una por trabajo.
export CUDA_VISIBLE_DEVICES=0

# El clon del clúster no trae .git: definir el commit del codigo copiado, si se conoce.
export GIT_COMMIT="${GIT_COMMIT:-no definido}"

cd "$HOME/proyecto_de_titulo"

echo "nodo: $(hostname)"
echo "fecha: $(date -Is)"
echo "semilla: $SEED | datos: $DATA_ROOT"
nvidia-smi

python -u scripts/train_m1_fase1.py --seed "$SEED" --data-root "$DATA_ROOT"
