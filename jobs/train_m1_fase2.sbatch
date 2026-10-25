#!/bin/bash
# Fase 2 de M1 (fine-tuning) para un candidato de profundidad de descongelado,
# solo train y validation. Parte del mejor checkpoint de una corrida de Fase 1.
# Uso, desde ~/proyecto_de_titulo en el nodo de login:
#   sbatch --export=ALL,CANDIDATE=block7_top,FASE1_RUN=runs/m1_fase1/seed7_<fecha>,GPU_ID=0 jobs/train_m1_fase2.sbatch
# Candidatos: block7_top, block6_top, block5_top. GPU_ID = 0 o 1 (hay dos V100).
# Con el commit del codigo copiado: GIT_COMMIT=<hash> sbatch --export=ALL,... jobs/train_m1_fase2.sbatch
#SBATCH -J m1_fase2
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

: "${CANDIDATE:?definir CANDIDATE (block7_top, block6_top o block5_top)}"
: "${FASE1_RUN:?definir FASE1_RUN (carpeta de la corrida de Fase 1)}"
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
echo "candidato: $CANDIDATE | fase 1: $FASE1_RUN | datos: $DATA_ROOT | GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

python -u scripts/train_m1_fase2.py --candidate "$CANDIDATE" --fase1-run "$FASE1_RUN" --data-root "$DATA_ROOT"
