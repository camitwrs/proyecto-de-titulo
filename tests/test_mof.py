import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.mof import CRITERIOS, ablation, comparar_por_semilla, mof, perturbar, signo

with open(PROJECT_ROOT / "reports" / "ahp_report.json", encoding="utf-8") as _f:
    _w = json.load(_f)["weights"]
# Pesos exactos del autovector (no los redondeados a 4 decimales del informe).
AHP = {"accuracy_global": _w["accuracy_global"], "recall_macro": _w["recall_macro"], "auc_macro_ovr": _w["auc_macro"]}


class TestMOF(unittest.TestCase):
    def test_perturbacion_conserva_cambio_relativo_y_suma_uno(self):
        validos, descartados = perturbar(AHP, 0.10)
        self.assertEqual(len(validos), 6)
        self.assertEqual(descartados, {})
        for nombre, vec in validos.items():
            self.assertAlmostEqual(sum(vec.values()), 1.0, places=9)
            self.assertGreaterEqual(min(vec.values()), 0)
        v = validos["recall_macro_mas10"]
        self.assertAlmostEqual(v["recall_macro"], AHP["recall_macro"] * 1.10)
        # Los otros dos mantienen su proporcion mutua.
        self.assertAlmostEqual(v["accuracy_global"] / v["auc_macro_ovr"], AHP["accuracy_global"] / AHP["auc_macro_ovr"])

    def test_perturbacion_no_admisible_y_peso_cero(self):
        _, descartados = perturbar({"accuracy_global": 0.95, "recall_macro": 0.05, "auc_macro_ovr": 0.0}, 0.10)
        self.assertIn("accuracy_global_mas10", descartados)  # 0.95 * 1.10 > 1
        self.assertEqual(descartados["auc_macro_ovr_mas10"], "peso_cero_sin_cambio")

    def test_ablation_coincide_con_los_vectores_declarados(self):
        v = ablation(AHP)
        self.assertAlmostEqual(v["sin_accuracy_global"]["recall_macro"], 0.7115, places=4)
        self.assertAlmostEqual(v["sin_recall_macro"]["accuracy_global"], 0.2885, places=4)
        self.assertAlmostEqual(v["sin_auc_macro_ovr"]["recall_macro"], 0.8588, places=4)
        for vec in v.values():
            self.assertAlmostEqual(sum(vec.values()), 1.0)

    def test_ablation_vs_yaml(self):
        import yaml
        with open(PROJECT_ROOT / "configs" / "pilot_m1_m4.yaml", encoding="utf-8") as f:
            declarado = yaml.safe_load(f)["evaluation"]["mof"]["ablation"]["variants"]
        calculado = ablation(AHP)
        for nombre_yaml, nombre in (("sin_accuracy_global", "sin_accuracy_global"),
                                    ("sin_recall_macro", "sin_recall_macro"),
                                    ("sin_auc_macro", "sin_auc_macro_ovr")):
            for k, j in (("w1", "accuracy_global"), ("w2", "recall_macro"), ("w3", "auc_macro_ovr")):
                self.assertAlmostEqual(declarado[nombre_yaml][k], calculado[nombre][j], places=4)

    def test_signo_con_tolerancia(self):
        self.assertEqual(signo(0.05, 1e-4), 1)
        self.assertEqual(signo(-0.05, 1e-4), -1)
        self.assertEqual(signo(5e-5, 1e-4), 0)

    def test_comparacion_por_semilla(self):
        m1 = {1: {"accuracy_global": 0.8, "recall_macro": 0.7, "auc_macro_ovr": 0.9},
              2: {"accuracy_global": 0.6, "recall_macro": 0.5, "auc_macro_ovr": 0.8}}
        m4 = {1: {"accuracy_global": 0.7, "recall_macro": 0.6, "auc_macro_ovr": 0.8},
              2: {"accuracy_global": 0.7, "recall_macro": 0.4, "auc_macro_ovr": 0.8}}
        w = {"accuracy_global": 0.2, "recall_macro": 0.6, "auc_macro_ovr": 0.2}
        r = comparar_por_semilla(m1, m4, w, 1e-4)
        self.assertAlmostEqual(r["por_semilla"][1]["diferencia"], 0.2 * 0.1 + 0.6 * 0.1 + 0.2 * 0.1)
        self.assertEqual(r["por_semilla"][1]["signo"], 1)
        # Con peso solo en accuracy, la semilla 2 invierte el signo.
        r2 = comparar_por_semilla(m1, m4, {"accuracy_global": 1.0, "recall_macro": 0.0, "auc_macro_ovr": 0.0}, 1e-4)
        self.assertEqual(r2["por_semilla"][2]["signo"], -1)
        with self.assertRaises(ValueError):
            comparar_por_semilla(m1, {1: m4[1]}, w, 1e-4)

    def test_mof_formula(self):
        m = {"accuracy_global": 0.5, "recall_macro": 0.5, "auc_macro_ovr": 0.5}
        self.assertAlmostEqual(mof(m, AHP), 0.5 * sum(AHP.values()))


if __name__ == "__main__":
    unittest.main()
