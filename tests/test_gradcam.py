import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.gradcam import calcular_gradcam, capa_objetivo, seleccionar_imagenes
from src.models.cnn_m4 import build_cnn_m4
from src.models.efficientnet_b4 import build_efficientnet_b4_m1

CONFIG = yaml.safe_load(open(PROJECT_ROOT / "configs" / "pilot_m1_m4.yaml", encoding="utf-8"))


def modelo_pequeno():
    entrada = tf.keras.Input((8, 8, 3))
    x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu", name="conv")(entrada)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.0)(x)
    return tf.keras.Model(entrada, tf.keras.layers.Dense(3, activation="softmax")(x))


class TestCalculo(unittest.TestCase):
    def test_coincide_con_cam_en_arquitectura_gap_lineal(self):
        # Con conv -> GAP -> Dense, Grad-CAM equivale a CAM: alpha_k = w_k / (h*w).
        tf.keras.utils.set_random_seed(0)
        m = modelo_pequeno()
        x = np.random.default_rng(0).random((2, 8, 8, 3), dtype=np.float32)
        clases = np.array([1, 2])
        _, crudo = calcular_gradcam(m, "conv", x, clases, 8)
        mapa = tf.keras.Model(m.inputs, m.get_layer("conv").output)(x).numpy()
        w = m.layers[-1].kernel.numpy()
        for i, c in enumerate(clases):
            esperado = np.maximum((mapa[i] * w[:, c]).sum(-1) / (8 * 8), 0)
            np.testing.assert_allclose(crudo[i], esperado, rtol=1e-4, atol=1e-6)

    def test_rango_forma_y_mapa_nulo(self):
        m = modelo_pequeno()
        x = np.random.default_rng(1).random((3, 8, 8, 3), dtype=np.float32)
        norm, _ = calcular_gradcam(m, "conv", x, np.array([0, 1, 2]), 16)
        self.assertEqual(norm.shape, (3, 16, 16))
        self.assertTrue(np.all(norm >= 0) and np.all(norm <= 1 + 1e-6))
        # Si todos los pesos de la capa final son 0, los gradientes son 0 y el mapa queda en 0.
        m.layers[-1].kernel.assign(tf.zeros_like(m.layers[-1].kernel))
        norm0, crudo0 = calcular_gradcam(m, "conv", x, np.array([0, 1, 2]), 16)
        self.assertEqual(float(norm0.max()), 0.0)
        self.assertEqual(float(crudo0.max()), 0.0)

    def test_capas_objetivo_de_m1_y_m4(self):
        cfg = {**CONFIG, "m1": {**CONFIG["m1"], "weights": None}}
        m1, _ = build_efficientnet_b4_m1(cfg)
        self.assertEqual(capa_objetivo(m1, "m1"), "efficientnetb4")
        m4 = build_cnn_m4(CONFIG)
        self.assertEqual(capa_objetivo(m4, "m4"), "activation_3")

    def test_funciona_en_m1_real(self):
        cfg = {**CONFIG, "m1": {**CONFIG["m1"], "weights": None}}
        m1, _ = build_efficientnet_b4_m1(cfg)
        x = np.random.default_rng(3).random((1, 380, 380, 3), dtype=np.float32) * 255
        norm, crudo = calcular_gradcam(m1, capa_objetivo(m1, "m1"), x, np.array([4]), 380)
        self.assertEqual(norm.shape, (1, 380, 380))
        self.assertEqual(crudo.shape[1:], (12, 12))
        self.assertTrue(np.isfinite(norm).all())

    def test_funciona_en_m4_real(self):
        m4 = build_cnn_m4(CONFIG)
        x = np.random.default_rng(2).random((2, 380, 380, 3), dtype=np.float32)
        norm, crudo = calcular_gradcam(m4, capa_objetivo(m4, "m4"), x, np.array([0, 5]), 380)
        self.assertEqual(norm.shape, (2, 380, 380))
        self.assertEqual(crudo.shape[1:], (48, 48))
        self.assertTrue(np.isfinite(norm).all())


class TestSeleccion(unittest.TestCase):
    @staticmethod
    def datos():
        rng = np.random.default_rng(0)
        filas = []
        for i in range(400):
            v = int(rng.integers(0, 8))
            p = v if rng.random() < 0.6 else int(rng.integers(0, 8))
            filas.append((f"ISIC_{i:07d}", v, p))
        return pd.DataFrame(filas, columns=["image", "clase_verdadera", "clase_predicha"])

    REGLAS = {"aciertos_por_clase": 2, "celdas_de_confusion": 3, "imagenes_por_celda": 2,
              "falsos_negativos_de": ["MEL", "BCC", "SCC"], "falsos_negativos_por_clase": 3}
    CLASES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]

    def test_es_reproducible_y_respeta_conteos(self):
        d = self.datos()
        a = seleccionar_imagenes(d, self.CLASES, self.REGLAS, 42)
        b = seleccionar_imagenes(d, self.CLASES, self.REGLAS, 42)
        pd.testing.assert_frame_equal(a, b)
        self.assertLessEqual((a["categoria"] == "acierto").sum(), 16)
        self.assertLessEqual((a["categoria"] == "confusion").sum(), 6)
        self.assertLessEqual((a["categoria"] == "falso_negativo").sum(), 9)

    def test_categorias_son_correctas(self):
        d = self.datos()
        s = seleccionar_imagenes(d, self.CLASES, self.REGLAS, 42)
        for _, f in s.iterrows():
            if f["categoria"] == "acierto":
                self.assertEqual(f["clase_verdadera"], f["clase_predicha"])
            else:
                self.assertNotEqual(f["clase_verdadera"], f["clase_predicha"])
            if f["categoria"] == "falso_negativo":
                self.assertIn(f["clase_verdadera"], ("MEL", "BCC", "SCC"))

    def test_semilla_distinta_cambia_la_muestra(self):
        d = self.datos()
        a = seleccionar_imagenes(d, self.CLASES, self.REGLAS, 42)
        b = seleccionar_imagenes(d, self.CLASES, self.REGLAS, 43)
        self.assertFalse(a.equals(b))


if __name__ == "__main__":
    unittest.main()
