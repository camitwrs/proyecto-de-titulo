import tensorflow as tf


def _build_head_layer(layer_cfg):
    tipo = layer_cfg["layer"]

    if tipo == "GlobalAveragePooling2D":
        return tf.keras.layers.GlobalAveragePooling2D()

    if tipo == "BatchNormalization":
        return tf.keras.layers.BatchNormalization()

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


def build_efficientnet_b4_m1(config):
    """Construye M1 (EfficientNet-B4 + cabeza piloto) leyendo la arquitectura
    directamente desde la seccion 'm1' de configs/pilot_m1_m4.yaml.

    Retorna (model, backbone): el modelo completo y una referencia directa al
    backbone EfficientNetB4, necesaria para congelar/descongelar capas por fase
    durante el entrenamiento en dos fases (ver Slide 12 de la presentacion de avance).
    """
    m1_cfg = config["m1"]
    input_shape = tuple(m1_cfg["input_shape"])

    backbone = tf.keras.applications.EfficientNetB4(
        include_top=m1_cfg.get("include_top", False),
        weights=m1_cfg.get("weights", "imagenet"),
        input_shape=input_shape,
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = backbone(inputs, training=False)
    for layer_cfg in m1_cfg["head"]:
        x = _build_head_layer(layer_cfg)(x)

    model = tf.keras.Model(inputs, x, name="efficientnet_b4_m1")

    return model, backbone


def build_efficientnet_b4(config):
    input_shape = tuple(config["input_shape"])
    num_classes = config["num_classes"]
    head_cfg = config["head"]

    base = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights=config.get("weights", "imagenet"),
        input_shape=input_shape,
    )

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(head_cfg["dropout"])(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation=head_cfg["classifier_activation"],
        dtype="float32",
    )(x)

    return tf.keras.Model(inputs, outputs, name="efficientnet_b4_smoke")
