#!/bin/bash
# Evalua las corridas finales (M1 y M4) con scripts/evaluar_modelo.py. Un solo trabajo recorre
# las semillas y los modelos pedidos, porque el cluster permite pocos trabajos enviados a la vez.
# Es el unico paso que lee el conjunto de test, y solo lo hace si se da CONFIRMO=si.
#
# Comprobacion mecanica sobre validation, con pocas imagenes (NO toca el test):
#   GIT_COMMIT=<hash> sbatch --export=ALL,PARTICION=validation,POR_CLASE=10,SEEDS=42,GPU_ID=0 jobs/evaluar_final.sbatch
# Evaluacion sobre test, una sola vez, con la configuracion y las diez corridas ya congeladas:
#   GIT_COMMIT=<hash> sbatch --export=ALL,PARTICION=test,CONFIRMO=si,GPU_ID=0 jobs/evaluar_final.sbatch
# Por defecto recorre las semillas 42 123 456 789 2026 y los modelos m1 y m4; se pueden acotar
# con SEEDS="42 123" y MODELOS="m4".
#SBATCH -J evaluar
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

PARTICION="${PARTICION:?definir PARTICION (validation o test)}"
SEEDS="${SEEDS:-42 123 456 789 2026}"
MODELOS="${MODELOS:-m1 m4}"
DATA_ROOT="${DATA_ROOT:-/work/$USER/isic}"

# Leer test exige confirmar de forma explicita que todo esta congelado.
EXTRA=()
if [ "$PARTICION" = "test" ]; then
    [ "${CONFIRMO:-}" = "si" ] || { echo "Para evaluar test hay que definir CONFIRMO=si"; exit 1; }
    EXTRA+=(--confirmo-config-congelada)
fi
[ -n "${POR_CLASE:-}" ] && EXTRA+=(--por-clase "$POR_CLASE")

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
echo "particion: $PARTICION | modelos: $MODELOS | semillas: $SEEDS | por clase: ${POR_CLASE:-todas} | commit: $GIT_COMMIT"
nvidia-smi

ultima() { ls -d "$@" | sort | tail -1; }

for MODELO in $MODELOS; do
    for SEED in $SEEDS; do
        echo "=== $MODELO semilla $SEED ==="
        if [ "$MODELO" = "m1" ]; then
            FASE1=$(ultima runs/final/m1_fase1/seed${SEED}_*)
            FASE2=$(ultima runs/final/m1_fase2/*_seed${SEED}_*)
            python -u scripts/evaluar_modelo.py --modelo m1 --fase1-run "$FASE1" --fase2-run "$FASE2" \
                --particion "$PARTICION" --data-root "$DATA_ROOT" "${EXTRA[@]}"
        else
            RUN_M4=$(ultima runs/final/m4/seed${SEED}_*)
            python -u scripts/evaluar_modelo.py --modelo m4 --run-m4 "$RUN_M4" \
                --particion "$PARTICION" --data-root "$DATA_ROOT" "${EXTRA[@]}"
        fi
    done
done
