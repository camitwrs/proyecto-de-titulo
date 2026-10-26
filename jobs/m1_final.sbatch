#!/bin/bash
# Corrida final de M1 para UNA semilla: Fase 1 (cabeza) y despues Fase 2 (ajuste fino) con la
# profundidad elegida en reports/seleccion_descongelado.json. Solo train y validation; el test
# se evalua aparte con scripts/evaluar_modelo.py. Usa la QoS short-gpu (maximo 4 h), que tiene mayor
# prioridad que normal-gpu segun el equipo del HPC; las dos fases suman unas 2 h segun los pilotos. Las dos fases van en un solo trabajo porque
# el cluster permite pocos trabajos enviados a la vez.
# Uso, desde ~/proyecto_de_titulo en el nodo de login:
#   GIT_COMMIT=<hash> sbatch --export=ALL,SEED=42,GPU_ID=0 jobs/m1_final.sbatch
# Para reanudar solo la Fase 2 tras un fallo, dar la carpeta de la Fase 1 ya completada:
#   GIT_COMMIT=<hash> sbatch --export=ALL,SEED=42,GPU_ID=0,FASE1_RUN=runs/final/m1_fase1/seed42_<fecha> jobs/m1_final.sbatch
#SBATCH -J m1_final
#SBATCH -p GPU
#SBATCH --qos=short-gpu
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH -o logs/%j_%x.out
#SBATCH -e logs/%j_%x.err

set -eo pipefail

: "${SEED:?definir SEED (42, 123, 456, 789 o 2026)}"
DATA_ROOT="${DATA_ROOT:-/work/$USER/isic}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tesis

# Sin esto TensorFlow no encuentra las librerias CUDA instaladas por pip.
NV_DIR=$(python -c "import nvidia; print(list(nvidia.__path__)[0])")
export LD_LIBRARY_PATH=$(ls -d "$NV_DIR"/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}

# Slurm no administra las GPU (Gres=null): las dos son visibles, se fija una por trabajo.
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

# El clon del cluster no trae .git: definir el commit del codigo copiado.
export GIT_COMMIT="${GIT_COMMIT:-no definido}"

cd "$HOME/proyecto_de_titulo"

CANDIDATO=$(python -c "import json; print(json.load(open('reports/seleccion_descongelado.json'))['selected'])")

echo "nodo: $(hostname)"
echo "fecha: $(date -Is)"
echo "semilla: $SEED | candidato de Fase 2: $CANDIDATO | datos: $DATA_ROOT | GPU: $CUDA_VISIBLE_DEVICES | commit: $GIT_COMMIT"
nvidia-smi

if [ -z "${FASE1_RUN:-}" ]; then
    python -u scripts/train_m1_fase1.py --seed "$SEED" --data-root "$DATA_ROOT" --output-root runs/final/m1_fase1
    # El script de la Fase 1 crea una carpeta nueva con la fecha; se toma la mas reciente de esta semilla.
    FASE1_RUN=$(ls -d runs/final/m1_fase1/seed${SEED}_* | sort | tail -1)
fi
echo "Fase 1 de origen: $FASE1_RUN"

python -u scripts/train_m1_fase2.py --candidate "$CANDIDATO" --fase1-run "$FASE1_RUN" \
    --data-root "$DATA_ROOT" --output-root runs/final/m1_fase2
