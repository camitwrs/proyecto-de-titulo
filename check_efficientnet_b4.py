import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
from tensorflow.keras import mixed_precision

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError("TensorFlow no detecta GPU.")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

mixed_precision.set_global_policy("mixed_float16")

print("TensorFlow:", tf.__version__)
print("GPU:", gpus)
print("Política de precisión:", mixed_precision.global_policy().name)

base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(380, 380, 3),
    pooling="avg",
)
base.trainable = False

inputs = tf.keras.Input(shape=(380, 380, 3))
x = base(inputs, training=False)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.30)(x)
outputs = tf.keras.layers.Dense(8, activation="softmax", dtype="float32")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

x_batch = tf.random.uniform((4, 380, 380, 3), dtype=tf.float32)
y_batch = tf.random.uniform((4,), minval=0, maxval=8, dtype=tf.int32)

loss, accuracy = model.train_on_batch(x_batch, y_batch)

print("Modelo construido correctamente.")
print("Batch de entrenamiento comprobado: 4 imágenes de 380x380.")
print("Loss:", float(loss))
print("Accuracy:", float(accuracy))
print("Salida esperada: 8 probabilidades por imagen.")
