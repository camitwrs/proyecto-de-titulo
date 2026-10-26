import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metricas import calcular_metricas, calcular_mof

CLASES = ["A", "B", "C"]


def _probs(pred, n_classes=3, confianza=0.8):
    """Probabilidades que ponen `confianza` en la clase predicha y reparten el resto."""
    p = np.full((len(pred), n_classes), (1 - confianza) / (n_classes - 1))
    p[np.arange(len(pred)), pred] = confianza
    return p


class TestMetricas(unittest.TestCase):
    def test_caso_calculado_a_mano(self):
        # matriz de confusion [[1,1,0],[0,2,0],[1,0,1]]
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]
        m = calcular_metricas(y_true, _probs(y_pred), CLASES)
        self.assertAlmostEqual(m["accuracy_global"], 4 / 6)
        self.assertAlmostEqual(m["recall_per_class"]["A"], 0.5)
        self.assertAlmostEqual(m["recall_per_class"]["B"], 1.0)
        self.assertAlmostEqual(m["recall_macro"], 2 / 3)
        self.assertAlmostEqual(m["precision_per_class"]["A"], 0.5)
        self.assertAlmostEqual(m["precision_per_class"]["B"], 2 / 3)
        self.assertAlmostEqual(m["precision_per_class"]["C"], 1.0)
        self.assertAlmostEqual(m["f1_per_class"]["B"], 0.8)
        self.assertEqual(m["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])

    def test_prediccion_perfecta(self):
        y = [0, 1, 2, 0, 1, 2]
        m = calcular_metricas(y, _probs(y, confianza=0.9), CLASES)
        for clave in ("accuracy_global", "recall_macro", "precision_macro", "f1_macro", "auc_macro_ovr"):
            self.assertAlmostEqual(m[clave], 1.0, msg=clave)

    def test_auc_invertido_es_cero(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        p = np.zeros((6, 3))
        for i, c in enumerate(y):
            p[i] = 0.1
            p[i, (c + 1) % 3] = 0.8
        m = calcular_metricas(y, p, CLASES)
        self.assertAlmostEqual(m["accuracy_global"], 0.0)
        self.assertLess(m["auc_macro_ovr"], 0.5)

    def test_clase_nunca_predicha_tiene_precision_cero(self):
        y_true = [0, 1, 2, 2]
        y_pred = [0, 1, 0, 1]
        m = calcular_metricas(y_true, _probs(y_pred), CLASES)
        self.assertEqual(m["precision_per_class"]["C"], 0.0)
        self.assertEqual(m["f1_per_class"]["C"], 0.0)

    def test_macro_no_depende_de_la_frecuencia_de_clase(self):
        # 8 de clase A siempre acertadas, 1 de clase B fallada: accuracy alta, recall macro baja.
        y_true = [0] * 8 + [1] + [2]
        y_pred = [0] * 8 + [0] + [2]
        m = calcular_metricas(y_true, _probs(y_pred), CLASES)
        self.assertAlmostEqual(m["accuracy_global"], 0.9)
        self.assertAlmostEqual(m["recall_macro"], 2 / 3)

    def test_rechaza_clase_ausente_y_forma_invalida(self):
        with self.assertRaises(ValueError):
            calcular_metricas([0, 1, 1], _probs([0, 1, 1]), CLASES)
        with self.assertRaises(ValueError):
            calcular_metricas([0, 1, 2], _probs([0, 1, 2])[:, :2], CLASES)

    def test_mof(self):
        m = {"accuracy_global": 0.6, "recall_macro": 0.5, "auc_macro_ovr": 0.9}
        w = {"accuracy_global": 0.1, "recall_macro": 0.6, "auc_macro_ovr": 0.3}
        self.assertAlmostEqual(calcular_mof(m, w), 0.06 + 0.30 + 0.27)
        with self.assertRaises(ValueError):
            calcular_mof(m, {"accuracy_global": 0.5, "recall_macro": 0.6, "auc_macro_ovr": 0.3})


if __name__ == "__main__":
    unittest.main()
