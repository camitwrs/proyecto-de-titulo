"""Resume las corridas finales (solo train y validation) de M1 y M4.

Lee los run_info.json de runs/final/ y reporta, por semilla, las epocas completadas, la mejor
epoca y la mejor val_loss; la media y la desviacion estandar muestral (ddof=1) entre semillas
de la mejor val_loss; y cuantas corridas terminaron por el techo de epocas en lugar de parar
por paciencia. La desviacion entre semillas es la medida de ruido que el informe se comprometio
a reportar. No usa el conjunto de test.

Solo usa la libreria estandar, para poder correr tanto en el cluster como en local:
  python3 scripts/resumen_corridas_finales.py
Escribe reports/resumen_corridas_finales.json.
"""

import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL = PROJECT_ROOT / "runs" / "final"
SALIDA = PROJECT_ROOT / "reports" / "resumen_corridas_finales.json"
# Una mejor epoca dentro de estas primeras epocas, en M4, sugiere que la seleccion tomo un pico temprano.
EPOCAS_TEMPRANAS = 10


def leer(patron):
    corridas = []
    for ruta in sorted(FINAL.glob(patron)):
        with open(ruta, "r", encoding="utf-8") as f:
            info = json.load(f)
        if info.get("status") != "completed":
            raise SystemExit(f"Corrida no completada: {ruta} (status={info.get('status')})")
        corridas.append(info)
    return sorted(corridas, key=lambda info: info["seed"])


def resumir(corridas):
    perdidas = [info["best_val_loss"] for info in corridas]
    return {
        "n": len(corridas),
        "semillas": [info["seed"] for info in corridas],
        "best_val_loss": perdidas,
        "media_best_val_loss": statistics.mean(perdidas),
        "desv_best_val_loss": statistics.stdev(perdidas) if len(perdidas) > 1 else None,
        "epocas_completadas": [info["epochs_completed"] for info in corridas],
        "mejor_epoca": [info["best_epoch"] for info in corridas],
        "techo_de_epocas": sorted({info["max_epochs"] for info in corridas}),
        "corridas_que_llegaron_al_techo": [
            info["seed"] for info in corridas if info["epochs_completed"] >= info["max_epochs"]
        ],
    }


def main():
    m1_fase1 = leer("m1_fase1/seed*/run_info.json")
    m1_fase2 = leer("m1_fase2/*_seed*/run_info.json")
    m4 = leer("m4/seed*/run_info.json")
    if not (m1_fase1 and m1_fase2 and m4):
        raise SystemExit("Faltan corridas finales en runs/final/")

    resumen = {
        "uses_test": False,
        "desviacion": "estandar muestral (ddof=1) entre semillas",
        "m1_fase1": resumir(m1_fase1),
        "m1_fase2_final": resumir(m1_fase2),
        "m4": resumir(m4),
    }
    resumen["m4"]["semillas_con_mejor_epoca_temprana"] = [
        info["seed"] for info in m4 if info["best_epoch"] <= EPOCAS_TEMPRANAS
    ]
    resumen["m4"]["epocas_tempranas_hasta"] = EPOCAS_TEMPRANAS

    peor_m1 = max(resumen["m1_fase2_final"]["best_val_loss"])
    mejor_m4 = min(resumen["m4"]["best_val_loss"])
    resumen["brecha"] = {
        "diferencia_de_medias_m4_menos_m1": resumen["m4"]["media_best_val_loss"]
        - resumen["m1_fase2_final"]["media_best_val_loss"],
        "mejor_m4_menos_peor_m1": mejor_m4 - peor_m1,
    }

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(SALIDA, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    json.dump(resumen, sys.stdout, indent=2, ensure_ascii=False)
    print(f"\nEscrito en {SALIDA}")


if __name__ == "__main__":
    main()
