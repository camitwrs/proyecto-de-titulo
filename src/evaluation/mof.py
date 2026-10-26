"""Analisis post-hoc de la MOF: perturbacion de pesos uno a la vez, ablation y
comparacion M1 vs M4 por semilla. Opera sobre metricas ya calculadas; no entrena,
no usa pesos del modelo y no decide nada con el test.

Las reglas replican lo declarado en el informe y en configs/pilot_m1_m4.yaml
(evaluation.mof): perturbacion relativa de +-10 % con redistribucion proporcional,
y ablation que quita una metrica y reparte su peso proporcionalmente.
"""

CRITERIOS = ("accuracy_global", "recall_macro", "auc_macro_ovr")


def mof(metricas, pesos):
    return sum(pesos[c] * metricas[c] for c in CRITERIOS)


def perturbar(pesos, delta=0.10, tol_suma=1e-9):
    """Perturba un peso a la vez en +-delta (relativo) y reparte el resto en proporcion.

    w_i' = (1 + d) w_i ;  w_j' = w_j [1 - (1 + d) w_i] / (1 - w_i)  para j != i.
    Devuelve {nombre: vector} con las perturbaciones admisibles y {nombre: motivo}
    con las que no lo son, sin recortar valores ni cambiar la regla.
    """
    validos, descartados = {}, {}
    for i in CRITERIOS:
        for d in (-delta, +delta):
            nombre = f"{i}_{'mas' if d > 0 else 'menos'}{int(round(abs(d) * 100))}"
            wi = pesos[i]
            nuevo = (1 + d) * wi
            if wi == 0:
                descartados[nombre] = "peso_cero_sin_cambio"
                continue
            if not (0 <= wi < 1) or not (0 <= nuevo <= 1):
                descartados[nombre] = "no_admisible"
                continue
            vec = {j: (nuevo if j == i else pesos[j] * (1 - nuevo) / (1 - wi)) for j in CRITERIOS}
            if abs(sum(vec.values()) - 1) > tol_suma or min(vec.values()) < 0:
                raise ValueError(f"Vector perturbado invalido en {nombre}: {vec}")
            validos[nombre] = vec
    return validos, descartados


def ablation(pesos):
    """Quita una metrica a la vez y reparte su peso proporcionalmente: w'_j = w_j / suma de los demas."""
    variantes = {}
    for k in CRITERIOS:
        resto = sum(pesos[j] for j in CRITERIOS if j != k)
        if resto <= 0:
            raise ValueError(f"No se puede quitar {k}: los pesos restantes suman 0")
        variantes[f"sin_{k}"] = {j: (0.0 if j == k else pesos[j] / resto) for j in CRITERIOS}
    return variantes


def signo(diferencia, tolerancia):
    if abs(diferencia) <= tolerancia:
        return 0
    return 1 if diferencia > 0 else -1


def comparar_por_semilla(metricas_m1, metricas_m4, pesos, tolerancia):
    """Para un vector de pesos, devuelve MOF de cada modelo, la diferencia M1 - M4 y su signo por semilla."""
    if set(metricas_m1) != set(metricas_m4):
        raise ValueError(f"Las semillas de M1 {sorted(metricas_m1)} y M4 {sorted(metricas_m4)} no coinciden")
    filas = {}
    for semilla in sorted(metricas_m1):
        a, b = mof(metricas_m1[semilla], pesos), mof(metricas_m4[semilla], pesos)
        filas[semilla] = {"mof_m1": a, "mof_m4": b, "diferencia": a - b, "signo": signo(a - b, tolerancia)}
    media_m1 = sum(f["mof_m1"] for f in filas.values()) / len(filas)
    media_m4 = sum(f["mof_m4"] for f in filas.values()) / len(filas)
    return {
        "por_semilla": filas,
        "media_m1": media_m1,
        "media_m4": media_m4,
        "orden_por_media": signo(media_m1 - media_m4, tolerancia),
    }
