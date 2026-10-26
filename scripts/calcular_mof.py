"""Aplica la MOF y su analisis de sensibilidad sobre metricas ya calculadas por
scripts/evaluar_modelo.py. No entrena, no carga modelos y no decide nada con el test.

Vectores analizados (reglas en configs/pilot_m1_m4.yaml, evaluation.mof):
- referencia: AHP y los escenarios C0 a C3;
- perturbacion uno a la vez de +-10 % sobre cada vector de referencia;
- ablation (quitar una metrica a la vez) sobre el vector AHP.

Por cada vector y semilla compara MOF(M1) - MOF(M4) y examina si su signo se
mantiene, se invierte o pasa a empate respecto de la referencia.

Uso:
    python scripts/calcular_mof.py --m1 runs/evaluacion/m1_seed42_test_* ... --m4 runs/evaluacion/m4_seed42_test_* ...
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.mof import CRITERIOS, ablation, comparar_por_semilla, perturbar

CONFIG_PATH = PROJECT_ROOT / "configs" / "pilot_m1_m4.yaml"
AHP_PATH = PROJECT_ROOT / "reports" / "ahp_report.json"


def cargar_evaluaciones(carpetas):
    """{semilla: metricas} de carpetas creadas por evaluar_modelo.py; todas deben ser la misma particion."""
    por_semilla, particiones = {}, set()
    for carpeta in carpetas:
        carpeta = Path(carpeta)
        with open(carpeta / "evaluacion_info.json", "r", encoding="utf-8") as f:
            info = json.load(f)
        with open(carpeta / "metricas.json", "r", encoding="utf-8") as f:
            metricas = json.load(f)
        if info["seed"] in por_semilla:
            raise SystemExit(f"Semilla {info['seed']} repetida entre las carpetas dadas")
        por_semilla[info["seed"]] = metricas
        particiones.add(info["particion"])
    return por_semilla, particiones


def vectores_de_referencia(config):
    with open(AHP_PATH, "r", encoding="utf-8") as f:
        ahp = json.load(f)["weights"]
    # El reporte AHP usa "auc_macro"; las metricas del modelo, "auc_macro_ovr".
    referencias = {"AHP": {"accuracy_global": ahp["accuracy_global"], "recall_macro": ahp["recall_macro"],
                           "auc_macro_ovr": ahp["auc_macro"]}}
    for nombre, w in config["evaluation"]["mof"]["sensitivity_scenarios"].items():
        referencias[nombre] = {"accuracy_global": w["w1"], "recall_macro": w["w2"], "auc_macro_ovr": w["w3"]}
    return referencias


def main():
    parser = argparse.ArgumentParser(description="MOF, sensibilidad y ablation sobre metricas ya calculadas")
    parser.add_argument("--m1", nargs="+", required=True, help="carpetas de evaluacion de M1 (una por semilla)")
    parser.add_argument("--m4", nargs="+", required=True, help="carpetas de evaluacion de M4 (una por semilla)")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "mof_resultados.json")
    args = parser.parse_args()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    reglas = config["evaluation"]["mof"]["analysis_rules"]
    tolerancia = reglas["tie_tolerance"]

    m1, part1 = cargar_evaluaciones(args.m1)
    m4, part4 = cargar_evaluaciones(args.m4)
    if part1 != part4 or len(part1) != 1:
        raise SystemExit(f"M1 y M4 deben evaluarse sobre la misma particion; recibido {part1} y {part4}")

    referencias = vectores_de_referencia(config)
    resultados, descartados = {}, {}

    def registrar(nombre, pesos, origen, tipo):
        analisis = comparar_por_semilla(m1, m4, pesos, tolerancia)
        resultados[nombre] = {"tipo": tipo, "origen": origen, "pesos": pesos, **analisis}

    for nombre, pesos in referencias.items():
        registrar(nombre, pesos, None, "referencia")
        validos, malos = perturbar(pesos, reglas["perturbation_relative_delta"], reglas["weight_sum_tolerance"])
        for sub, vec in validos.items():
            registrar(f"{nombre}:{sub}", vec, nombre, "perturbacion")
        for sub, motivo in malos.items():
            descartados[f"{nombre}:{sub}"] = motivo
    for sub, vec in ablation(referencias["AHP"]).items():
        registrar(f"AHP:{sub}", vec, "AHP", "ablation")

    # Estabilidad: el signo de cada derivado frente al de su referencia, semilla por semilla.
    for nombre, r in resultados.items():
        if r["origen"] is None:
            continue
        ref = resultados[r["origen"]]["por_semilla"]
        r["estabilidad"] = {
            str(s): ("se_mantiene" if f["signo"] == ref[s]["signo"]
                     else "empate" if f["signo"] == 0
                     else "pasa_de_empate" if ref[s]["signo"] == 0
                     else "se_invierte")
            for s, f in r["por_semilla"].items()
        }
        r["orden_por_media_igual_a_referencia"] = r["orden_por_media"] == resultados[r["origen"]]["orden_por_media"]

    salida = {
        "particion": next(iter(part1)),
        "tolerancia_empate": tolerancia,
        "semillas": sorted(m1),
        "vectores": resultados,
        "perturbaciones_no_admisibles": descartados,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    csv_path = args.output.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["vector", "tipo", "semilla", "mof_m1", "mof_m4", "diferencia", "signo"])
        for nombre, r in resultados.items():
            for s, fila in r["por_semilla"].items():
                w.writerow([nombre, r["tipo"], s, f"{fila['mof_m1']:.6f}", f"{fila['mof_m4']:.6f}",
                            f"{fila['diferencia']:.6f}", fila["signo"]])
    print(f"{len(resultados)} vectores, {len(m1)} semillas ({salida['particion']}). Guardado en {args.output} y {csv_path.name}")
    if descartados:
        print(f"Perturbaciones no admisibles: {descartados}")


if __name__ == "__main__":
    main()
