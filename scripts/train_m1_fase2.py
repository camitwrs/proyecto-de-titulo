"""Entrena la Fase 2 de M1 (fine-tuning) para UN candidato de profundidad de
descongelado, partiendo del mejor checkpoint de una corrida de la Fase 1.
Usa solo train/validation; el conjunto de test nunca se carga.

Cada corrida guarda en runs/m1_fase2/<candidato>_seed<S>_<fecha>/:
config_snapshot.yaml, class_weights.json, history.csv, fase2_mejor.weights.h5
y run_info.json (candidato, corrida de Fase 1 de origen, parametros entrenables,
mejor epoca, commit).

La eleccion entre candidatos la hace scripts/seleccionar_descongelado.py con la
regla declarada en configs/pilot_m1_m4.yaml (seleccion_descongelado).

Uso en el HPC: ver jobs/train_m1_fase2.sbatch
"""

import argparse
import json
import os
import platform
import shutil
import sys
import time
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
    build_callbacks,
    compute_class_weights,
    git_commit,
    load_config,
    subsample_per_class,
    write_json,
)
from src.data.isic2019 import build_dataset
from src.models.efficientnet_b4 import aplicar_descongelado_fase2, build_efficientnet_b4_m1
from src.utils.reproducibility import set_global_seed

FASE1_CHECKPOINT = "fase1_mejor.weights.h5"


def parse_args():
    parser = argparse.ArgumentParser(description="Fase 2 de M1 para un candidato (train/validation, sin test)")
    parser.add_argument("--candidate", required=True, help="block7_top, block6_top o block5_top")
    parser.add_argument("--fase1-run", type=Path, required=True,
                        help="carpeta de la corrida de Fase 1 (contiene fase1_mejor.weights.h5 y run_info.json)")
    parser.add_argument("--epochs", type=int, default=None, help="maximo de epocas (por defecto, el del YAML)")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT,
                        help="carpeta que contiene data/isic2019/... (por defecto, la raiz del proyecto)")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "m1_fase2")
    parser.add_argument("--train-per-class", type=int, default=None, help="submuestra de train por clase (solo pruebas)")
    parser.add_argument("--val-per-class", type=int, default=None, help="submuestra de validation por clase (solo pruebas)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(CONFIG_PATH)

    fase_cfg = config["m1"]["training"]["fase_2"]
    if args.candidate not in fase_cfg["candidates"]:
        raise SystemExit(f"Candidato desconocido: {args.candidate}. Opciones: {list(fase_cfg['candidates'])}")
    prefixes = fase_cfg["candidates"][args.candidate]["prefixes"]

    fase1_ckpt = args.fase1_run / FASE1_CHECKPOINT
    fase1_info_path = args.fase1_run / "run_info.json"
    if not fase1_ckpt.exists() or not fase1_info_path.exists():
        raise SystemExit(f"Falta {FASE1_CHECKPOINT} o run_info.json en {args.fase1_run}")
    with open(fase1_info_path, "r", encoding="utf-8") as f:
        fase1_info = json.load(f)
    if fase1_info.get("status") != "completed":
        raise SystemExit(f"La corrida de Fase 1 no esta completada (status={fase1_info.get('status')})")

    # La Fase 2 usa la misma semilla que la corrida de Fase 1 de la que parte.
    seed = fase1_info["seed"]
    max_epochs = args.epochs if args.epochs is not None else fase_cfg["max_epochs"]
    class_order = config["dataset"]["classes"]["order"]
    batch_size = config["common"]["batch_size"]
    image_size = config["m1"]["input_shape"][0]

    run_dir = args.output_root / f"{args.candidate}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(CONFIG_PATH, run_dir / "config_snapshot.yaml")

    gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    run_info = {
        "status": "running",
        "fase": 2,
        "candidate": args.candidate,
        "unfreeze_prefixes": prefixes,
        "seed": seed,
        "max_epochs": max_epochs,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "devices": gpus if gpus else ["CPU"],
        "host": platform.node(),
        "data_root": str(args.data_root),
        "fase1_run": str(args.fase1_run),
        "fase1_best_epoch": fase1_info.get("best_epoch"),
        "fase1_best_val_loss": fase1_info.get("best_val_loss"),
        "subsample": {"train_per_class": args.train_per_class, "val_per_class": args.val_per_class},
    }
    write_json(run_dir / "run_info.json", run_info)
    print(f"Carpeta de la corrida: {run_dir}")
    print(f"Candidato: {args.candidate} | Fase 1 de origen: {args.fase1_run}")
    print(f"Dispositivos: {run_info['devices']}")

    started = time.time()
    try:
        set_global_seed(seed)

        df = pd.read_csv(PARTITION_CSV)
        df_train = df[df["split"] == "train"].reset_index(drop=True)
        df_val = df[df["split"] == "validation"].reset_index(drop=True)
        df_train = subsample_per_class(df_train, args.train_per_class, seed)
        df_val = subsample_per_class(df_val, args.val_per_class, seed)
        run_info["n_train"] = len(df_train)
        run_info["n_validation"] = len(df_val)
        print(f"train: {len(df_train)} imagenes | validation: {len(df_val)} imagenes")

        first_image = args.data_root / df_train["image_path_relative"].iloc[0]
        if not first_image.exists():
            raise FileNotFoundError(
                f"No se encuentra {first_image}. Usa --data-root con la carpeta que contiene data/isic2019/..."
            )

        class_weights = compute_class_weights(df_train, class_order)
        write_json(run_dir / "class_weights.json",
                   {class_order[idx]: weight for idx, weight in class_weights.items()})

        num_classes = len(class_order)
        ds_train = build_dataset(
            df_train, args.data_root, image_size=image_size, batch_size=batch_size,
            normalize_fn=lambda image: image, training=True, num_classes=num_classes,
            augmentation_cfg=config["preprocessing"]["augmentation"], shuffle_seed=seed,
        )
        ds_val = build_dataset(
            df_val, args.data_root, image_size=image_size, batch_size=batch_size,
            normalize_fn=lambda image: image, training=False, num_classes=num_classes,
        )

        model, backbone = build_efficientnet_b4_m1(config)
        model.load_weights(str(fase1_ckpt))
        run_info["layers_unfrozen"] = aplicar_descongelado_fase2(backbone, prefixes)

        opt_cfg = config["common"]["optimizer_defaults"]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=fase_cfg["optimizer"]["learning_rate"],
                beta_1=opt_cfg["beta_1"], beta_2=opt_cfg["beta_2"], epsilon=opt_cfg["epsilon"],
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=config["common"]["loss"]["label_smoothing"]
            ),
            metrics=["accuracy"],
        )
        run_info["params_total"] = int(model.count_params())
        run_info["params_trainable"] = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
        print(f"parametros totales: {run_info['params_total']:,} | entrenables: {run_info['params_trainable']:,}")

        history = model.fit(
            ds_train, validation_data=ds_val, epochs=max_epochs,
            class_weight=class_weights,
            callbacks=build_callbacks(config, run_dir, checkpoint_name="fase2_mejor.weights.h5"),
            verbose=2,
        )

        val_loss = history.history["val_loss"]
        run_info["epochs_completed"] = len(val_loss)
        run_info["best_epoch"] = int(np.argmin(val_loss)) + 1
        run_info["best_val_loss"] = float(np.min(val_loss))
        run_info["status"] = "completed"
    except BaseException as exc:
        run_info["status"] = "failed"
        run_info["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        run_info["finished_at"] = datetime.now().isoformat(timespec="seconds")
        run_info["elapsed_seconds"] = round(time.time() - started, 1)
        write_json(run_dir / "run_info.json", run_info)
        print(f"Estado: {run_info['status']} | run_info.json en {run_dir}")


if __name__ == "__main__":
    main()
