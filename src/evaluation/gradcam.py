"""Grad-CAM (Selvaraju et al., 2017) para M1 y M4.

Para una clase c, calcula el gradiente del puntaje de la clase ANTES del softmax
respecto del mapa de activaciones A^k de la ultima capa convolucional, promedia
ese gradiente en el espacio para obtener los pesos alpha_k y combina:
    L = ReLU( sum_k alpha_k A^k )
El mapa se reescala a la resolucion de entrada. Es una visualizacion cualitativa:
no demuestra causalidad ni validez clinica.
"""

import numpy as np
import pandas as pd
import tensorflow as tf


def capa_objetivo(model, modelo):
    """Nombre de la ultima capa convolucional.

    M1: el backbone anidado; su salida es el mapa de activaciones final de
        EfficientNet-B4 (top_activation), antes del pooling.
    M4: la ultima activacion ReLU, que sigue a la cuarta convolucion y precede
        al ultimo MaxPooling.
    """
    if modelo == "m1":
        anidados = [l for l in model.layers if isinstance(l, tf.keras.Model)]
        if len(anidados) != 1:
            raise ValueError(f"Se esperaba un unico backbone anidado en M1; hay {len(anidados)}")
        return anidados[0].name
    activaciones = [l for l in model.layers if isinstance(l, tf.keras.layers.Activation)]
    if not activaciones:
        raise ValueError("M4 no tiene capas Activation")
    return activaciones[-1].name


def calcular_gradcam(model, nombre_capa, imagenes, clases, tam_salida):
    """Devuelve (mapas_normalizados, mapas_crudos) para cada imagen y su clase.

    imagenes: (N, H, W, 3) ya normalizadas como las recibe el modelo.
    clases: (N,) indice de la clase a explicar en cada imagen.
    mapas_crudos: ReLU(sum alpha_k A^k) a la resolucion de la capa, sin normalizar.
    mapas_normalizados: reescalados a tam_salida x tam_salida y divididos por su
        maximo (0 si el mapa es todo cero), para visualizar en [0, 1].
    """
    ultima = model.layers[-1]            # Dense con softmax
    capas = [l for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
    if capas[-1] is not ultima or nombre_capa not in [l.name for l in capas]:
        raise ValueError("La capa objetivo debe estar en la cadena lineal de capas del modelo")

    imagenes = tf.convert_to_tensor(imagenes, dtype=tf.float32)
    clases = tf.convert_to_tensor(clases, dtype=tf.int32)
    with tf.GradientTape() as tape:
        # M1 y M4 son cadenas lineales de capas; se avanza capa por capa (en modo de inferencia)
        # para tener el mapa de la capa objetivo dentro de la cinta. No se usa layer.output porque
        # en un backbone anidado devuelve el tensor interno y no el del grafo exterior.
        h = imagenes
        mapa = None
        for capa in capas[:-1]:
            h = capa(h, training=False)
            if capa.name == nombre_capa:
                mapa = h
                tape.watch(mapa)
        # Puntaje antes del softmax: z = h @ W + b; la capa final ya aplica softmax.
        logits = tf.matmul(h, ultima.kernel) + ultima.bias
        puntaje = tf.gather(logits, clases, batch_dims=1)
    grads = tape.gradient(puntaje, mapa)
    alpha = tf.reduce_mean(grads, axis=(1, 2))                                  # (N, K)
    crudo = tf.nn.relu(tf.reduce_sum(alpha[:, None, None, :] * mapa, axis=-1))   # (N, h, w)

    grande = tf.image.resize(crudo[..., None], (tam_salida, tam_salida), method="bilinear")[..., 0]
    maximo = tf.reduce_max(grande, axis=(1, 2), keepdims=True)
    normalizado = tf.where(maximo > 0, grande / tf.maximum(maximo, 1e-12), tf.zeros_like(grande))
    return normalizado.numpy(), crudo.numpy()


def seleccionar_imagenes(pred, class_order, reglas, semilla):
    """Seleccion declarada y reproducible de imagenes a visualizar.

    pred: DataFrame con 'image', 'clase_verdadera', 'clase_predicha' (indices) y
        las columnas prob_<clase>, como escribe scripts/evaluar_modelo.py.
    reglas: bloque evaluation.gradcam.seleccion del YAML.
    Categorias: aciertos por clase, las celdas de confusion mas frecuentes y los
    falsos negativos de las clases indicadas. Dentro de cada grupo se muestrea al
    azar con la semilla dada; no se mira ninguna imagen para elegirla.
    """
    rng = np.random.default_rng(semilla)

    def tomar(df, n):
        if len(df) <= n:
            return df.sort_values("image")
        idx = rng.choice(len(df), size=n, replace=False)
        return df.sort_values("image").iloc[sorted(idx)]

    pred = pred.sort_values("image").reset_index(drop=True)
    elegidas = []

    for c, nombre in enumerate(class_order):
        grupo = pred[(pred["clase_verdadera"] == c) & (pred["clase_predicha"] == c)]
        for _, fila in tomar(grupo, reglas["aciertos_por_clase"]).iterrows():
            elegidas.append((fila["image"], "acierto", nombre, nombre))

    errores = pred[pred["clase_verdadera"] != pred["clase_predicha"]]
    celdas = (errores.groupby(["clase_verdadera", "clase_predicha"]).size()
              .reset_index(name="n").sort_values(["n", "clase_verdadera", "clase_predicha"], ascending=[False, True, True])
              .head(reglas["celdas_de_confusion"]))
    for _, celda in celdas.iterrows():
        grupo = errores[(errores["clase_verdadera"] == celda["clase_verdadera"]) &
                        (errores["clase_predicha"] == celda["clase_predicha"])]
        for _, fila in tomar(grupo, reglas["imagenes_por_celda"]).iterrows():
            elegidas.append((fila["image"], "confusion", class_order[int(fila["clase_verdadera"])],
                             class_order[int(fila["clase_predicha"])]))

    for nombre in reglas["falsos_negativos_de"]:
        c = class_order.index(nombre)
        grupo = errores[errores["clase_verdadera"] == c]
        for _, fila in tomar(grupo, reglas["falsos_negativos_por_clase"]).iterrows():
            elegidas.append((fila["image"], "falso_negativo", nombre, class_order[int(fila["clase_predicha"])]))

    return pd.DataFrame(elegidas, columns=["image", "categoria", "clase_verdadera", "clase_predicha"]).drop_duplicates(
        subset=["image", "categoria"]).reset_index(drop=True)
