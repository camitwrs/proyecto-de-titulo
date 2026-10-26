"""Elige la profundidad de descongelado de la Fase 2 de M1 con la regla
declarada en configs/pilot_m1_m4.yaml (m1.training.fase_2.seleccion_descongelado):
menor val_loss entre los candidatos; si varios quedan a menos de tie_tolerance
del minimo, gana el que tiene menos parametros entrenables.

Solo lee los run_info.json de las corridas de Fase 2 con la semilla de seleccion.
No usa el conjunto de test.

Uso (todos los candidatos del YAML, o solo algunos con --candidatos):
    python scripts/seleccionar_descongelado.py --runs-root runs/m1_fase2
    python scripts/seleccionar_descongelado.py --candidatos block7_top block6_top block5_top
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "pilot_m1_m4.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "seleccion_descongelado.json"


def load_runs(runs_root, selection_seed, candidates):
    """Una corrida completada por candidato; si hay varias, falla para no elegir a mano."""
    found = {name: [] for name in candidates}
    for info_path in sorted(Path(runs_root).glob("*/run_info.json")):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        if info.get("fase") != 2 or info.get("seed") != selection_seed or info.get("status") != "completed":
            continue
        if info.get("subsample", {}).get("train_per_class") is not None:
            continue
        if info["candidate"] in found:
            found[info["candidate"]].append((info_path.parent, info))
    for name, runs in found.items():
        if len(runs) != 1:
            raise SystemExit(
                f"Se esperaba exactamente 1 corrida completada de '{name}' con semilla {selection_seed}; "
                f"hay {len(runs)}. Deja solo la corrida valida en {runs_root}."
            )
    return {name: runs[0] for name, runs in found.items()}


def select(runs, tie_tolerance):
    min_loss = min(info["best_val_loss"] for _, info in runs.values())
    tied = [name for name, (_, info) in runs.items() if info["best_val_loss"] - min_loss <= tie_tolerance]
    winner = min(tied, key=lambda name: runs[name][1]["params_trainable"])
    return winner, tied


def main():
    parser = argparse.ArgumentParser(description="Seleccion de profundidad de descongelado (solo validation)")
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs" / "m1_fase2")
    parser.add_argument("--candidatos", nargs="+", default=None,
                        help="candidatos a comparar (por defecto, todos los del YAML)")
    args = parser.parse_args()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        fase_cfg = yaml.safe_load(f)["m1"]["training"]["fase_2"]
    rule = fase_cfg["seleccion_descongelado"]

    candidatos = args.candidatos if args.candidatos else list(fase_cfg["candidates"])
    desconocidos = [c for c in candidatos if c not in fase_cfg["candidates"]]
    if desconocidos:
        raise SystemExit(f"Candidatos que no estan en el YAML: {desconocidos}")

    runs = load_runs(args.runs_root, rule["selection_seed"], candidatos)
    winner, tied = select(runs, rule["tie_tolerance"])

    print(f"Semilla de seleccion: {rule['selection_seed']} | tolerancia de empate: {rule['tie_tolerance']}")
    for name, (run_dir, info) in runs.items():
        print(f"  {name}: val_loss={info['best_val_loss']:.4f} (epoca {info['best_epoch']}) "
              f"| entrenables={info['params_trainable']:,} | {run_dir.name}")
    print(f"Candidatos dentro de la tolerancia: {tied}")
    print(f"Seleccionado: {winner}")

    report = {
        "selection_seed": rule["selection_seed"],
        "criterion": rule["criterion"],
        "tie_tolerance": rule["tie_tolerance"],
        "tie_break": rule["tie_break"],
        "uses_test": False,
        "candidates": {
            name: {
                "run": run_dir.name,
                "best_val_loss": info["best_val_loss"],
                "best_epoch": info["best_epoch"],
                "params_trainable": info["params_trainable"],
                "commit": info["commit"],
            }
            for name, (run_dir, info) in runs.items()
        },
        "within_tolerance": tied,
        "selected": winner,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Reporte guardado en {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
