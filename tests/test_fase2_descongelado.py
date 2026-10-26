import sys
import unittest
from pathlib import Path

import tensorflow as tf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.efficientnet_b4 import aplicar_descongelado_fase2, build_efficientnet_b4_m1

CONFIG_PATH = PROJECT_ROOT / "configs" / "pilot_m1_m4.yaml"
HEAD_PARAMS = 464648

# Parametros entrenables del backbone por candidato (verificados contra Keras).
EXPECTED_BACKBONE_PARAMS = {
    "block7_top": 5253748,
    "block6_top": 13836088,
    "block5_top": 16119004,
    "block4_top": 17162774,
    "block3_top": 17355112,
}


class TestDescongeladoFase2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)
        cls.config["m1"]["weights"] = None
        cls.candidates = cls.config["m1"]["training"]["fase_2"]["candidates"]

    def _build(self, name):
        model, backbone = build_efficientnet_b4_m1(self.config)
        aplicar_descongelado_fase2(backbone, self.candidates[name]["prefixes"])
        return model, backbone

    def test_trainable_param_counts(self):
        for name, backbone_params in EXPECTED_BACKBONE_PARAMS.items():
            with self.subTest(candidate=name):
                model, _ = self._build(name)
                trainable = sum(int(tf.size(w).numpy()) for w in model.trainable_weights)
                self.assertEqual(trainable, backbone_params + HEAD_PARAMS)

    def test_backbone_batchnorm_stays_frozen(self):
        for name in self.candidates:
            with self.subTest(candidate=name):
                _, backbone = self._build(name)
                for layer in backbone.layers:
                    if isinstance(layer, tf.keras.layers.BatchNormalization):
                        self.assertFalse(layer.trainable, layer.name)

    def test_only_selected_blocks_are_trainable(self):
        _, backbone = self._build("block7_top")
        for layer in backbone.layers:
            if layer.trainable:
                self.assertTrue(layer.name.startswith(("block7", "top_")), layer.name)

    def test_candidates_are_nested(self):
        counts = {}
        for name in EXPECTED_BACKBONE_PARAMS:
            _, backbone = self._build(name)
            counts[name] = sum(int(tf.size(w).numpy()) for w in backbone.trainable_weights)
        self.assertLess(counts["block7_top"], counts["block6_top"])
        self.assertLess(counts["block6_top"], counts["block5_top"])
        self.assertLess(counts["block5_top"], counts["block4_top"])
        self.assertLess(counts["block4_top"], counts["block3_top"])


if __name__ == "__main__":
    unittest.main()
