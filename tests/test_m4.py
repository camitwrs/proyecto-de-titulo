import sys
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_m4 import build_cnn_m4

CONFIG_PATH = PROJECT_ROOT / "configs" / "pilot_m1_m4.yaml"
# Cuenta directa de la especificacion de configs/pilot_m1_m4.yaml (informe, Tabla de M4).
EXPECTED_PARAMS = 457704


class TestM4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)
        cls.model = build_cnn_m4(cls.config)

    def test_total_params(self):
        self.assertEqual(self.model.count_params(), EXPECTED_PARAMS)

    def test_all_params_trainable_or_bn_stats(self):
        # Todo es entrenable salvo las estadisticas moviles de BatchNormalization.
        n_bn = sum(1 for l in self.model.layers if isinstance(l, tf.keras.layers.BatchNormalization))
        self.assertEqual(n_bn, 4)
        non_trainable = sum(int(tf.size(w).numpy()) for w in self.model.non_trainable_weights)
        bn_stats = 2 * sum(l.gamma.shape[0] for l in self.model.layers if isinstance(l, tf.keras.layers.BatchNormalization))
        self.assertEqual(non_trainable, bn_stats)

    def test_output_is_valid_softmax(self):
        x = np.random.default_rng(0).random((2, 380, 380, 3), dtype=np.float32)
        y = self.model(x, training=False).numpy()
        self.assertEqual(y.shape, (2, 8))
        np.testing.assert_allclose(y.sum(axis=1), 1.0, atol=1e-4)

    def test_output_dtype_float32(self):
        self.assertEqual(self.model.output.dtype, "float32")

    def test_conv_layers_have_no_bias(self):
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                self.assertFalse(layer.use_bias, layer.name)


if __name__ == "__main__":
    unittest.main()
