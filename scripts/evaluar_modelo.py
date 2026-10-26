"""Evalua un modelo ya entrenado (M1 o M4) sobre una particion y guarda las
predicciones y las metricas. Es el unico script que puede leer el conjunto de
test, por eso esa particion exige una confirmacion explicita.

Reglas del protocolo que aplica:
- Se evalua desde el archivo de pesos persistido en disco, recargado en un
  modelo nuevo, no desde el modelo que quedo en memoria al entrenar.
- Para M1 el checkpoint final es el de menor val_loss entre la Fase 1 y la
  Fase 2; si empatan exactamente, gana la Fase 1 (primera en el orden global).
- El test no sirve para elegir checkpoint ni hiperparametros: esta prohibido
  usarlo para decidir nada. Este script solo mide.

Cada evaluacion deja en runs/evaluacion/<modelo>_seed<S>_<particion>_<fecha>/:
predicciones.csv, metricas.json y evaluacion_info.json (checkpoint, su SHA-256,
corridas de origen, commit, versiones).

Comprobacion mecanica sin tocar el test (usa validation y pocas imagenes):
    python scripts/evaluar_modelo.py --modelo m4 --run-m4 <carpeta> --particion validation --por-clase 10
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import tensorflow as tf

from scripts.train_m1_fase1 import (
    CONFIG_PATH,
    PARTITION_CSV,
    git_commit,
    load_config,
    subsample_per_class,
    write_json,
)
from src.data.isic2019 import build_dataset
from src.evaluation.metricas import calcular_metricas
from src.models.cnn_m4 import build_cnn_m4
from src.models.efficientnet_b4 import build_efficientnet_b4_m1


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluacion de M1 o M4 desde un checkpoint persistido")
    parser.add_argument("--modelo", required=True, choices=["m1", "m4"])
    parser.add_argument("--fase1-run", type=Path, help="M1: carpeta de la corrida de Fase 1")
    parser.add_argument("--fase2-run", type=Path, help="M1: carpeta de la corrida de Fase 2")
    parser.add_argument("--run-m4", type=Path, help="M4: carpeta de la corrida")
    parser.add_argument("--particion", default="test", choices=["test", "validation"])
    parser.add_argument("--confirmo-config-congelada", action="store_true",
                        help="obligatorio para leer test: la configuracion y las corridas ya estan congeladas")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "evaluacion")
    parser.add_argument("--por-clase", type=int, default=None,
                        help="submuestra por clase, solo para comprobar el script con validation")
    return parser.parse_args()


def leer_run_info(run_dir):
    path = Path(run_dir) / "run_info.json"
    if not path.exists():
        raise SystemExit(f"No existe {path}")
    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)
    if info.get("status") != "completed":
        raise SystemExit(f"La corrida {run_dir} no esta completada (status={info.get('status')})")
    return info


def sha256_de(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def elegir_checkpoint(args):
    """Devuelve (ruta del checkpoint, seed, detalle de la eleccion)."""
    if args.modelo == "m4":
        if not args.run_m4:
            raise SystemExit("M4 requiere --run-m4")
        info = leer_run_info(args.run_m4)
        return args.run_m4 / "m4_mejor.weights.h5", info["seed"], {"origen": str(args.run_m4)}

    if not (args.fase1_run and args.fase2_run):
        raise SystemExit("M1 requiere --fase1-run y --fase2-run")
    info1 = leer_run_info(args.fase1_run)
    info2 = leer_run_info(args.fase2_run)
    if info1["seed"] != info2["seed"]:
        raise SystemExit(f"Las semillas no coinciden: fase 1 = {info1['seed']}, fase 2 = {info2['seed']}")
    # Menor val_loss entre fases; un empate exacto lo gana la Fase 1 (primera en el orden global).
    if info2["best_val_loss"] < info1["best_val_loss"]:
        ckpt, fase = args.fase2_run / "fase2_mejor.weights.h5", 2
    else:
        ckpt, fase = args.fase1_run / "fase1_mejor.weights.h5", 1
    detalle = {
        "fase_elegida": fase,
        "fase1_run": str(args.fase1_run), "fase1_best_val_loss": info1["best_val_loss"],
        "fase2_run": str(args.fase2_run), "fase2_best_val_loss": info2["best_val_loss"],
        "fase2_candidato": info2.get("candidate"),
    }
    return ckpt, info1["seed"], detalle


def construir_modelo(modelo, config):
    """Devuelve (modelo sin pesos cargados, tamano de entrada, funcion de normalizacion)."""
    if modelo == "m1":
        cfg = copy.deepcopy(config)
        cfg["m1"]["weights"] = None
        model, _ = build_efficientnet_b4_m1(cfg)
        return model, config["m1"]["input_shape"][0], (lambda image: image)
    return build_cnn_m4(config), config["m4"]["input_shape"][0], (lambda image: image / 255.0)


def main():
    args = parse_args()
    if args.particion == "test" and not args.confirmo_config_congelada:
        raise SystemExit(
            "Leer test exige --confirmo-config-congelada: confirma que la configuracion, la profundidad "
            "de descongelado y las corridas estan congeladas. El test no se usa para decidir nada."
        )
    if args.particion == "test" and args.por_clase is not None:
        raise SystemExit("--por-clase es solo para comprobar el script con validation; no se permite con test")

    config = load_config(CONFIG_PATH)
    class_order = config["dataset"]["classes"]["order"]
    batch_size = config["common"]["batch_size"]

    ckpt, seed, detalle = elegir_checkpoint(args)
    if not ckpt.exists():
        raise SystemExit(f"No existe el checkpoint {ckpt}")

    df = pd.read_csv(PARTITION_CSV)
    df = df[df["split"] == args.particion].reset_index(drop=True)
    df = subsample_per_class(df, args.por_clase, seed)
    first_image = args.data_root / df["image_path_relative"].iloc[0]
    if not first_image.exists():
        raise SystemExit(f"No se encuentra {first_image}. Usa --data-root con la carpeta que contiene data/isic2019/...")

    # Modelo nuevo, con los pesos leidos del archivo; los pesos ImageNet no hacen falta porque se sobrescriben.
    model, image_size, normalizar = construir_modelo(args.modelo, config)

    model.load_weights(str(ckpt))

    # Sin aumento y sin barajar: el orden de las predicciones coincide con el del DataFrame.
    ds = build_dataset(
        df, args.data_root, image_size=image_size, batch_size=batch_size,
        normalize_fn=normalizar, training=False, num_classes=len(class_order),
    )
    y_prob = model.predict(ds, verbose=2)
    if len(y_prob) != len(df):
        raise RuntimeError(f"Se esperaban {len(df)} predicciones y se obtuvieron {len(y_prob)}")

    metricas = calcular_metricas(df["class_index"].to_numpy(), y_prob, class_order)

    run_dir = args.output_root / f"{args.modelo}_seed{seed}_{args.particion}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    pred = pd.DataFrame(y_prob, columns=[f"prob_{c}" for c in class_order])
    pred.insert(0, "clase_verdadera", df["class_index"].to_numpy())
    pred.insert(0, "image", df["image"].to_numpy())
    pred.insert(2, "clase_predicha", y_prob.argmax(axis=1))
    pred.to_csv(run_dir / "predicciones.csv", index=False)
    write_json(run_dir / "metricas.json", metricas)
    write_json(run_dir / "evaluacion_info.json", {
        "modelo": args.modelo,
        "particion": args.particion,
        "seed": seed,
        "n_muestras": len(df),
        "subsample_por_clase": args.por_clase,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_de(ckpt),
        "eleccion": detalle,
        "commit": git_commit(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "dispositivos": [d.name for d in tf.config.list_physical_devices("GPU")] or ["CPU"],
        "fecha": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"Evaluacion guardada en {run_dir}")
    print(f"accuracy_global={metricas['accuracy_global']:.4f} | recall_macro={metricas['recall_macro']:.4f} "
          f"| auc_macro_ovr={metricas['auc_macro_ovr']:.4f}")


if __name__ == "__main__":
    main()
