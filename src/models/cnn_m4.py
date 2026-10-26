import tensorflow as tf


def _build_conv_block(block_cfg):
    """Un bloque de M4: Conv2D sin bias -> BatchNormalization -> ReLU -> MaxPooling2D -> Dropout."""
    pool_cfg = block_cfg["pooling"]
    return [
        tf.keras.layers.Conv2D(
            filters=block_cfg["filters"],
            kernel_size=tuple(block_cfg["kernel_size"]),
            strides=block_cfg["strides"],
            padding=block_cfg["padding"],
            use_bias=block_cfg["use_bias"],
            kernel_initializer=block_cfg["kernel_initializer"],
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(block_cfg["activation"]),
        tf.keras.layers.MaxPooling2D(
            pool_size=tuple(pool_cfg["pool_size"]),
            strides=pool_cfg["strides"],
            padding=pool_cfg["padding"],
        ),
        tf.keras.layers.Dropout(block_cfg["dropout"]),
    ]


def _build_head_layer(layer_cfg):
    tipo = layer_cfg["layer"]

    if tipo == "GlobalAveragePooling2D":
        return tf.keras.layers.GlobalAveragePooling2D()

    if tipo == "Dropout":
        return tf.keras.layers.Dropout(layer_cfg["rate"])

    if tipo == "Dense":
        regularizer_cfg = layer_cfg.get("kernel_regularizer")
        kernel_regularizer = None
        if regularizer_cfg is not None:
            if regularizer_cfg["type"] == "L2":
                kernel_regularizer = tf.keras.regularizers.L2(regularizer_cfg["value"])
            else:
                raise ValueError(f"Regularizador no soportado: {regularizer_cfg['type']}")

        return tf.keras.layers.Dense(
            units=layer_cfg["units"],
            activation=layer_cfg["activation"],
            kernel_initializer=layer_cfg.get("kernel_initializer", "glorot_uniform"),
            kernel_regularizer=kernel_regularizer,
            dtype=layer_cfg.get("dtype"),
        )

    raise ValueError(f"Tipo de capa de cabeza no soportado: {tipo}")


def build_cnn_m4(config):
    """Construye M4 (CNN desde cero) leyendo la arquitectura directamente desde
    la seccion 'm4' de configs/pilot_m1_m4.yaml. Sin pesos preentrenados; recibe
    imagenes en [0, 1] (la division entre 255 la hace el pipeline de datos)."""
    m4_cfg = config["m4"]
    inputs = tf.keras.Input(shape=tuple(m4_cfg["input_shape"]))

    x = inputs
    for block_cfg in m4_cfg["blocks"]:
        for layer in _build_conv_block(block_cfg):
            x = layer(x)

    for layer_cfg in m4_cfg["head"]:
        x = _build_head_layer(layer_cfg)(x)

    return tf.keras.Model(inputs, x, name="cnn_m4")
