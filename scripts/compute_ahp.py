"""Calcula el vector de prioridades AHP (autovector principal) y su
consistencia (CI, CR) a partir de la matriz de comparacion por pares
registrada en configs/pilot_m1_m4.yaml.

No elicita juicios ni los modifica: solo reproduce, de forma auditable y
reproducible, el calculo que produjo los pesos y el CR ya reportados en el
informe y en el propio YAML. Sirve para verificar esos numeros y para
recalcular si los juicios de la matriz cambian tras la revision bajo
supervision del proyecto de titulo.

Por que un script y no un calculo manual:
el autovector principal de una matriz 3x3 exige resolver una ecuacion
caracteristica cubica; hacerlo a mano es largo, propenso a error de
redondeo y no reproducible si un juicio cambia. Un script produce el mismo
numero cada vez, con precision de maquina, y puede volver a ejecutarse
en el momento sin rehacer el algebra desde cero.
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "pilot_m1_m4.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "ahp_report.json"

RANDOM_INDEX = {3: 0.58}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def matrix_from_config(ahp_cfg):
    order = ahp_cfg["criteria_order"]
    raw = ahp_cfg["pairwise_matrix"]
    n = len(order)
    A = np.zeros((n, n), dtype=float)
    for i, row_key in enumerate(order):
        for j, col_key in enumerate(order):
            A[i, j] = raw[row_key][col_key]
    return A, order


def principal_eigenvector(A):
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argmax(eigvals.real)
    lam_max = float(eigvals[idx].real)
    w = eigvecs[:, idx].real
    w = w / w.sum()
    return w, lam_max


def consistency(lam_max, n):
    ci = (lam_max - n) / (n - 1)
    ri = RANDOM_INDEX[n]
    cr = ci / ri
    return ci, ri, cr


def main():
    config = load_config(CONFIG_PATH)
    ahp_cfg = config["evaluation"]["ahp"]

    A, order = matrix_from_config(ahp_cfg)
    n = A.shape[0]

    w, lam_max = principal_eigenvector(A)
    ci, ri, cr = consistency(lam_max, n)
    accepted = cr <= 0.10

    print("Matriz de comparacion por pares (orden: %s)" % ", ".join(order))
    print(A)
    print()
    print("lambda_max = %.4f" % lam_max)
    print("CI = %.4f" % ci)
    print("RI = %.2f" % ri)
    print("CR = %.4f -> %s" % (cr, "aceptada" if accepted else "NO aceptada, revisar juicios"))
    print()
    print("Vector de prioridades:")
    for crit, wi in zip(order, w):
        print(f"  {crit}: {wi:.4f}")

    expected = ahp_cfg.get("weights")
    if expected:
        expected_vec = np.array([expected[f"w{i+1}_{c}"] for i, c in enumerate(order)])
        max_diff = float(np.max(np.abs(expected_vec - w)))
        print()
        print(f"Diferencia maxima vs. pesos ya registrados en el YAML: {max_diff:.6f}")

    report = {
        "criteria_order": order,
        "pairwise_matrix": A.tolist(),
        "lambda_max": lam_max,
        "ci": ci,
        "ri": ri,
        "cr": cr,
        "accepted": accepted,
        "weights": {c: float(wi) for c, wi in zip(order, w)},
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReporte guardado en {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
