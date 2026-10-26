"""Metricas de evaluacion de M1 y M4 sobre las ocho clases de ISIC 2019.

Todas se calculan a partir de las clases verdaderas y las probabilidades que
entrega el modelo; no entrenan ni modifican nada. Las metricas macro son el
promedio simple de las ocho clases, sin ponderar por su frecuencia: por eso
reflejan el desempeno en las clases minoritarias (DF, VASC) igual que en NV.

Convencion para clases sin predicciones: si una clase nunca es predicha, su
precision (y su F1) vale 0, no se omite ni se imputa. Se declara asi para que
el promedio macro no oculte esa falla.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score


def calcular_metricas(y_true, y_prob, class_order):
    """Devuelve un diccionario serializable con las metricas del YAML.

    y_true: clases verdaderas como enteros (N,).
    y_prob: probabilidades por clase (N, K), cada fila suma 1.
    class_order: nombres de clase en el orden de los indices.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n_classes = len(class_order)

    if y_prob.ndim != 2 or y_prob.shape != (len(y_true), n_classes):
        raise ValueError(f"y_prob debe tener forma ({len(y_true)}, {n_classes}); recibio {y_prob.shape}")
    if not np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("Las filas de y_prob deben sumar 1")
    ausentes = sorted(set(range(n_classes)) - set(np.unique(y_true).tolist()))
    if ausentes:
        raise ValueError(f"Clases sin ejemplos en y_true, el AUC no esta definido: {[class_order[i] for i in ausentes]}")

    y_pred = y_prob.argmax(axis=1)
    labels = list(range(n_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    auc_macro = roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr", labels=labels)

    return {
        "n_muestras": int(len(y_true)),
        "accuracy_global": float((y_pred == y_true).mean()),
        "precision_per_class": {c: float(v) for c, v in zip(class_order, precision)},
        "recall_per_class": {c: float(v) for c, v in zip(class_order, recall)},
        "f1_per_class": {c: float(v) for c, v in zip(class_order, f1)},
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "auc_macro_ovr": float(auc_macro),
        "confusion_matrix": cm.tolist(),
    }


def calcular_mof(metricas, pesos):
    """MOF = w1*accuracy_global + w2*recall_macro + w3*auc_macro, post-hoc.

    pesos: {"accuracy_global": w1, "recall_macro": w2, "auc_macro_ovr": w3}; deben sumar 1.
    """
    claves = ("accuracy_global", "recall_macro", "auc_macro_ovr")
    if set(pesos) != set(claves):
        raise ValueError(f"pesos debe tener exactamente las claves {claves}")
    if not np.isclose(sum(pesos.values()), 1.0, atol=1e-3):
        raise ValueError(f"Los pesos deben sumar 1; suman {sum(pesos.values()):.4f}")
    return float(sum(pesos[c] * metricas[c] for c in claves))
