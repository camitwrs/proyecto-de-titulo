import tensorflow as tf


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
