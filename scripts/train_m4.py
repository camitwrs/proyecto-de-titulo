"""Entrena M4 (CNN desde cero, una sola fase) con train/validation. El conjunto
de test nunca se carga ni se usa.

Cada corrida guarda en runs/m4/seed<S>_<fecha>/: config_snapshot.yaml,
class_weights.json, history.csv, m4_mejor.weights.h5 y run_info.json (estado,
versiones, dispositivo, commit, mejor epoca).

Uso local (validacion mecanica, muestra minima):
    python scripts/train_m4.py --train-per-class 30 --val-per-class 10 --epochs 1

Uso en el HPC: ver jobs/train_m4.sbatch
"""

import argparse
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
from src.models.cnn_m4 import build_cnn_m4
from src.utils.reproducibility import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento de M4 (train/validation, sin test)")
    parser.add_argument("--seed", type=int, default=None, help="semilla de entrenamiento (por defecto, la del YAML)")
    parser.add_argument("--epochs", type=int, default=None, help="maximo de epocas (por defecto, el del YAML)")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT,
                        help="carpeta que contiene data/isic2019/... (por defecto, la raiz del proyecto)")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "m4")
    parser.add_argument("--train-per-class", type=int, default=None, help="submuestra de train por clase (solo pruebas)")
    parser.add_argument("--val-per-class", type=int, default=None, help="submuestra de validation por clase (solo pruebas)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(CONFIG_PATH)

    seed = args.seed if args.seed is not None else config["seed"]
    train_cfg = config["m4"]["training"]
    max_epochs = args.epochs if args.epochs is not None else train_cfg["max_epochs"]
    class_order = config["dataset"]["classes"]["order"]
    batch_size = config["common"]["batch_size"]
    image_size = config["m4"]["input_shape"][0]

    run_dir = args.output_root / f"seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(CONFIG_PATH, run_dir / "config_snapshot.yaml")

    gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    run_info = {
        "status": "running",
        "modelo": "M4",
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
        "subsample": {"train_per_class": args.train_per_class, "val_per_class": args.val_per_class},
    }
    write_json(run_dir / "run_info.json", run_info)
    print(f"Carpeta de la corrida: {run_dir}")
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

        # M4 recibe imagenes en [0, 1]: se divide entre 255 despues del aumento.
        def normalizar_m4(image):
            return image / 255.0

        num_classes = len(class_order)
        ds_train = build_dataset(
            df_train, args.data_root, image_size=image_size, batch_size=batch_size,
            normalize_fn=normalizar_m4, training=True, num_classes=num_classes,
            augmentation_cfg=config["preprocessing"]["augmentation"], shuffle_seed=seed,
        )
        ds_val = build_dataset(
            df_val, args.data_root, image_size=image_size, batch_size=batch_size,
            normalize_fn=normalizar_m4, training=False, num_classes=num_classes,
        )

        model = build_cnn_m4(config)
        opt_cfg = config["common"]["optimizer_defaults"]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=train_cfg["optimizer"]["learning_rate"],
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
            callbacks=build_callbacks(config, run_dir, checkpoint_name="m4_mejor.weights.h5"),
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
