"""Genera visualizaciones Grad-CAM de un modelo ya evaluado, con la seleccion
declarada en configs/pilot_m1_m4.yaml (evaluation.gradcam). No entrena ni evalua:
lee las predicciones que dejo scripts/evaluar_modelo.py y recarga el checkpoint.

Deja en <evaluacion>/gradcam/: una figura por imagen (original, mapa de la clase
predicha y, en errores, mapa de la clase verdadera) y seleccion.json con el
criterio, las imagenes, el modelo, el checkpoint (con su SHA-256) y la capa
analizada. Las figuras son evidencia cualitativa limitada, no explicacion causal
ni validacion clinica.

Uso:
    python scripts/generar_gradcam.py --evaluacion runs/evaluacion/m1_seed42_test_<fecha> --confirmo-config-congelada
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from scripts.evaluar_modelo import construir_modelo, sha256_de
from scripts.train_m1_fase1 import CONFIG_PATH, PARTITION_CSV, load_config
from src.data.isic2019 import load_image
from src.evaluation.gradcam import calcular_gradcam, capa_objetivo, seleccionar_imagenes


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM con la seleccion declarada")
    parser.add_argument("--evaluacion", type=Path, required=True, help="carpeta creada por evaluar_modelo.py")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="ruta del checkpoint si la registrada no existe en este equipo; debe tener el mismo SHA-256")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--confirmo-config-congelada", action="store_true",
                        help="obligatorio si la evaluacion es sobre test")
    args = parser.parse_args()

    with open(args.evaluacion / "evaluacion_info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    if info["particion"] == "test" and not args.confirmo_config_congelada:
        raise SystemExit("La evaluacion es sobre test: confirma con --confirmo-config-congelada. "
                         "Grad-CAM solo ilustra, no decide nada.")

    config = load_config(CONFIG_PATH)
    reglas = config["evaluation"]["gradcam"]
    if info["seed"] != reglas["visualized_seed"]:
        raise SystemExit(f"Se declaro visualizar la semilla {reglas['visualized_seed']}; esta evaluacion es de la {info['seed']}")
    class_order = config["dataset"]["classes"]["order"]

    ckpt = args.checkpoint or Path(info["checkpoint"])
    if not ckpt.exists():
        raise SystemExit(f"No existe el checkpoint {ckpt}; indica --checkpoint")
    if sha256_de(ckpt) != info["checkpoint_sha256"]:
        raise SystemExit("El SHA-256 del checkpoint no coincide con el de la evaluacion")

    pred = pd.read_csv(args.evaluacion / "predicciones.csv")
    seleccion = seleccionar_imagenes(pred, class_order, reglas["selection"], reglas["selection"]["seed"])

    model, image_size, normalizar = construir_modelo(info["modelo"], config)
    model.load_weights(str(ckpt))
    capa = capa_objetivo(model, info["modelo"])

    rutas = pd.read_csv(PARTITION_CSV).set_index("image")["image_path_relative"]
    salida = args.evaluacion / "gradcam"
    salida.mkdir(exist_ok=True)
    columnas_prob = [f"prob_{c}" for c in class_order]
    pred = pred.set_index("image")

    registro = []
    for _, fila in seleccion.iterrows():
        nombre = fila["image"]
        original = load_image(str(args.data_root / rutas[nombre]), image_size)
        x = normalizar(original)[None]
        verdadera = class_order.index(fila["clase_verdadera"])
        predicha = class_order.index(fila["clase_predicha"])
        clases_a_explicar = [predicha] if verdadera == predicha else [predicha, verdadera]

        mapas = [calcular_gradcam(model, capa, x, np.array([c]), image_size)[0][0] for c in clases_a_explicar]
        fig, ejes = plt.subplots(1, 1 + len(mapas), figsize=(4 * (1 + len(mapas)), 4.2))
        ejes[0].imshow(original.numpy().astype("uint8"))
        ejes[0].set_title(f"{nombre}\nverdadera: {fila['clase_verdadera']}", fontsize=9)
        for eje, c, mapa in zip(ejes[1:], clases_a_explicar, mapas):
            eje.imshow(original.numpy().astype("uint8"))
            eje.imshow(mapa, cmap="jet", alpha=0.4, vmin=0, vmax=1)
            etiqueta = "predicha" if c == predicha else "verdadera"
            eje.set_title(f"Grad-CAM, clase {etiqueta}: {class_order[c]}\np = {pred.loc[nombre, columnas_prob[c]]:.3f}", fontsize=9)
        for eje in ejes:
            eje.axis("off")
        carpeta = salida / fila["categoria"]
        carpeta.mkdir(exist_ok=True)
        fig.savefig(carpeta / f"{nombre}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        registro.append({**fila.to_dict(), "clases_explicadas": [class_order[c] for c in clases_a_explicar]})

    with open(salida / "seleccion.json", "w", encoding="utf-8") as f:
        json.dump({
            "criterio": reglas["selection"],
            "metodo": reglas["method"],
            "modelo": info["modelo"],
            "semilla": info["seed"],
            "particion": info["particion"],
            "checkpoint": str(ckpt),
            "checkpoint_sha256": info["checkpoint_sha256"],
            "capa": capa,
            "imagenes": registro,
        }, f, indent=2, ensure_ascii=False)
    print(f"{len(registro)} figuras en {salida} (capa: {capa})")


if __name__ == "__main__":
    main()
