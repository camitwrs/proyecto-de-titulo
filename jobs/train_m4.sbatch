#!/bin/bash
# Entrenamiento de M4 (CNN desde cero), solo train y validation.
# Uso, desde ~/proyecto_de_titulo en el nodo de login:
#   mkdir -p logs && sbatch jobs/train_m4.sbatch
# Otra semilla, ruta de datos o carpeta de salida (OUTPUT_ROOT, p. ej. runs/final/m4):
#   sbatch --export=ALL,SEED=123,DATA_ROOT=/work/usuario/isic jobs/train_m4.sbatch
#SBATCH -J m4
#SBATCH -p GPU
#SBATCH --qos=normal-gpu
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
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
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

# El clon del clúster no trae .git: definir el commit del codigo copiado, si se conoce.
export GIT_COMMIT="${GIT_COMMIT:-no definido}"

cd "$HOME/proyecto_de_titulo"

echo "nodo: $(hostname)"
echo "fecha: $(date -Is)"
echo "semilla: $SEED | datos: $DATA_ROOT"
nvidia-smi

OUT_ARGS=()
[ -n "${OUTPUT_ROOT:-}" ] && OUT_ARGS=(--output-root "$OUTPUT_ROOT")

python -u scripts/train_m4.py --seed "$SEED" --data-root "$DATA_ROOT" "${OUT_ARGS[@]}"
