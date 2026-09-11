import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import tensorflow as tf

from src.models.efficientnet_b4 import build_efficientnet_b4
from src.utils.reproducibility import set_global_seed

CONFIG_PATH = PROJECT_ROOT / "configs" / "smoke_test_b4.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "smoke_test_b4_report.json"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def describe_device():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return f"GPU: {gpus[0].name}"
    return "CPU"


def run_smoke_test(config):
    checks = {}
    seed = config["seed"]
    set_global_seed(seed)

    batch_size = config["batch_size"]
    input_shape = tuple(config["input_shape"])
    num_classes = config["num_classes"]

    model = build_efficientnet_b4(config)
    checks["model_build"] = True

    x_batch = tf.random.uniform((batch_size, *input_shape), dtype=tf.float32)
    labels = tf.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tf.int32)
    y_batch = tf.one_hot(labels, depth=num_classes)

    y_pred = model(x_batch, training=False)
    output_shape = list(y_pred.shape)
    checks["forward_pass"] = output_shape == [batch_size, num_classes]

    row_sums = tf.reduce_sum(y_pred, axis=1).numpy().tolist()
    checks["softmax_valid"] = bool(
        all(abs(s - 1.0) < 1e-3 for s in row_sums)
        and tf.reduce_all(tf.math.is_finite(y_pred))
    )

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["optimizer"]["learning_rate"])

    with tf.GradientTape() as tape:
        y_pred_train = model(x_batch, training=True)
        loss = loss_fn(y_batch, y_pred_train)

    loss_value = float(loss.numpy())
    checks["finite_loss"] = bool(tf.math.is_finite(loss))

    trainable_vars = model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    finite_flags = [bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients if g is not None]
    checks["finite_gradients"] = len(finite_flags) > 0 and all(finite_flags)

    grads_and_vars = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
    try:
        optimizer.apply_gradients(grads_and_vars)
        checks["optimizer_step"] = True
    except Exception:
        checks["optimizer_step"] = False

    total_params = int(sum(np.prod(v.shape) for v in model.variables))
    trainable_params = int(sum(np.prod(v.shape) for v in trainable_vars))

    status = "PASS" if all(checks.values()) else "FAIL"

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "seed": seed,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "device": describe_device(),
        "model": config["model"],
        "weights": config["weights"],
        "input_shape": [batch_size, *input_shape],
        "output_shape": output_shape,
        "num_classes": num_classes,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "loss": loss_value,
        "probability_row_sums": row_sums,
        "checks": checks,
    }


def main():
    config = load_config(CONFIG_PATH)
    report = run_smoke_test(config)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for name, passed in report["checks"].items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print(f"\nStatus: {report['status']}")
    print(f"Report written to: {REPORT_PATH}")

    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
