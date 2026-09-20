import tensorflow as tf


def load_image(path, image_size):
    """Decodifica un JPEG y lo redimensiona a (image_size, image_size).

    Retorna float32 en rango [0, 255] (sin normalizar); cada modelo (M1/M4)
    aplica su propia normalizacion segun configs/pilot_m1_m4.yaml.
    """
    raw = tf.io.read_file(path)
    image = tf.io.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    return tf.cast(image, tf.float32)


def build_augmenter(augmentation_cfg):
    """Construye la funcion de aumento de datos segun
    configs/pilot_m1_m4.yaml -> preprocessing.augmentation.

    Orden fijo: flip -> rotacion -> zoom isotropico -> contraste, cada una
    aplicada con su propia probabilidad independiente. Solo debe usarse sobre
    el split de entrenamiento.
    """
    ops = augmentation_cfg["operations"]
    clip_min, clip_max = augmentation_cfg["clip_range_after_augmentation"]

    rot_cfg = ops["rotation"]
    rango_rotacion = max(abs(rot_cfg["degrees"][0]), abs(rot_cfg["degrees"][1])) / 360.0
    capa_rotacion = tf.keras.layers.RandomRotation(
        factor=rango_rotacion, fill_mode=rot_cfg["fill_mode"]
    )

    zoom_cfg = ops["isotropic_zoom"]
    capa_zoom = tf.keras.layers.RandomZoom(
        height_factor=tuple(zoom_cfg["factor"]),
        width_factor=tuple(zoom_cfg["factor"]),
        fill_mode=zoom_cfg["fill_mode"],
    )

    contraste_min, contraste_max = ops["contrast"]["factor"]

    def augmentar(image):
        if tf.random.uniform([]) < ops["horizontal_flip"]["probability"]:
            image = tf.image.flip_left_right(image)

        if tf.random.uniform([]) < rot_cfg["probability"]:
            image = capa_rotacion(tf.expand_dims(image, 0), training=True)[0]

        if tf.random.uniform([]) < zoom_cfg["probability"]:
            image = capa_zoom(tf.expand_dims(image, 0), training=True)[0]

        if tf.random.uniform([]) < ops["contrast"]["probability"]:
            image = tf.image.random_contrast(image, contraste_min, contraste_max)

        return tf.clip_by_value(image, clip_min, clip_max)

    return augmentar


def build_dataset(
    df,
    project_root,
    image_size,
    batch_size,
    normalize_fn,
    training,
    num_classes,
    augmentation_cfg=None,
    shuffle_seed=42,
):
    """Construye un tf.data.Dataset a partir de un DataFrame con columnas
    'image_path_relative' y 'class_index' (ver reports/splits/isic2019/).

    `normalize_fn` aplica la normalizacion especifica de cada modelo (M1: sin
    division adicional; M4: division por 255), despues de resize/augmentation.

    Las etiquetas se entregan codificadas en one-hot (`num_classes` columnas),
    ya que M1 y M4 usan `CategoricalCrossentropy` segun configs/pilot_m1_m4.yaml
    (no `SparseCategoricalCrossentropy`, que esperaria el entero sin codificar).
    """
    rutas = [str(project_root / p) for p in df["image_path_relative"]]
    etiquetas = df["class_index"].to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))

    if training:
        dataset = dataset.shuffle(buffer_size=len(df), seed=shuffle_seed, reshuffle_each_iteration=True)

    def _cargar(path, label):
        image = load_image(path, image_size)
        return image, label

    dataset = dataset.map(_cargar, num_parallel_calls=tf.data.AUTOTUNE)

    if training and augmentation_cfg is not None:
        augmentar = build_augmenter(augmentation_cfg)

        def _augmentar(image, label):
            return augmentar(image), label

        dataset = dataset.map(_augmentar, num_parallel_calls=tf.data.AUTOTUNE)

    def _normalizar(image, label):
        return normalize_fn(image), label

    dataset = dataset.map(_normalizar, num_parallel_calls=tf.data.AUTOTUNE)

    def _one_hot(image, label):
        return image, tf.one_hot(label, depth=num_classes)

    dataset = dataset.map(_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
