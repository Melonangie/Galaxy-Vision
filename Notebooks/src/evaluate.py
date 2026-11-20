"""Métricas de evaluación estándar del proyecto (ver README.md, sección Métricas de evaluación).

Nunca reportar solo accuracy. Pensado para reutilizarse tanto con los
baselines de sklearn como, en fases futuras, con la CNN/ResNet.

Incluye además las pruebas para comparar variantes de imágenes con muestras
chicas (ver README.md, sección Elegir la variante de imágenes):
`predictability_test` responde "¿esta representación predice este atributo
mejor que el azar?" (fuga norte/sur, atajo de la máscara, temprana/tardía), y
`bootstrap_auc_ci` / `paired_bootstrap_auc_diff` dan intervalos de confianza
para no declarar ganadora a una variante por ruido.
"""

from typing import Dict, List, Sequence

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    permutation_test_score,
)

from src.config import (
    PROBE_BOOTSTRAP_N,
    RANDOM_SEED,
    SHORTCUT_CV_FOLDS,
    SHORTCUT_CV_REPEATS,
    SHORTCUT_N_PERMUTATIONS,
)


def evaluate_binary_classifier(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, object]:
    """Calcula accuracy, precision/recall/F1 por clase, matriz de confusión y AUC-ROC.

    Args:
        y_true: etiquetas verdaderas, codificadas 0/1 (`src.config.LABEL_TO_INT`).
        y_pred: etiquetas predichas, codificadas 0/1.
        y_proba: probabilidad predicha de la clase positiva (1 = espiral).

    Returns:
        Diccionario con `accuracy`, `report` (salida de `classification_report`
        como dict), `confusion_matrix` y `roc_auc`.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def predictability_test(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = SHORTCUT_CV_FOLDS,
    n_repeats: int = SHORTCUT_CV_REPEATS,
    n_permutations: int = SHORTCUT_N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """¿Predice `X` el atributo binario `y` mejor que el azar?

    AUC-ROC con validación cruzada estratificada repetida (estimación estable
    en muestras chicas) y p-valor por permutación: se repite la validación
    cruzada con `y` barajada para ver qué AUC sale solo por azar con este
    tamaño de muestra.

    Args:
        estimator: clasificador de sklearn con `predict_proba` o
            `decision_function` (se clona en cada fold).
        X: matriz de features `(n_muestras, n_features)`.
        y: etiquetas 0/1.
        n_splits: folds de la validación cruzada.
        n_repeats: repeticiones con particiones distintas.
        n_permutations: permutaciones para el p-valor (0 lo omite).
        seed: semilla de particiones y permutaciones.

    Returns:
        `auc_media`, `auc_std` (entre folds y repeticiones) y `p_valor`
        (NaN si `n_permutations == 0`).
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    p_value = np.nan
    if n_permutations:
        _, _, p_value = permutation_test_score(
            estimator, X, y,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
            scoring="roc_auc", n_permutations=n_permutations, random_state=seed, n_jobs=-1,
        )
    return {"auc_media": float(scores.mean()), "auc_std": float(scores.std()), "p_valor": float(p_value)}


def out_of_fold_proba(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = SHORTCUT_CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Probabilidad de la clase positiva para cada muestra, predicha sin verla.

    Con la misma `seed`, las particiones son idénticas para cualquier `X` con
    las mismas `y`: así las predicciones de dos variantes se pueden comparar
    galaxia a galaxia (`paired_bootstrap_auc_diff`).

    Args:
        estimator: clasificador de sklearn con `predict_proba`.
        X: matriz de features.
        y: etiquetas 0/1.
        n_splits: folds de la validación cruzada.
        seed: semilla de las particiones.

    Returns:
        Arreglo `(n_muestras,)` con la probabilidad de `y == 1`.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(estimator, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def _bootstrap_indices(y: np.ndarray, n_boot: int, seed: int):
    """Genera remuestreos con reemplazo que contienen ambas clases."""
    rng = np.random.default_rng(seed)
    n = len(y)
    produced = 0
    while produced < n_boot:
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) == 2:
            produced += 1
            yield idx


def bootstrap_auc_ci(
    y: np.ndarray,
    proba: np.ndarray,
    n_boot: int = PROBE_BOOTSTRAP_N,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """AUC-ROC con intervalo de confianza bootstrap (percentiles).

    Args:
        y: etiquetas 0/1.
        proba: probabilidad predicha de la clase positiva (p. ej. de
            `out_of_fold_proba`).
        n_boot: número de remuestreos.
        alpha: 1 - nivel de confianza (0.05 = intervalo del 95%).
        seed: semilla del remuestreo.

    Returns:
        `auc`, `ic_inf` e `ic_sup`.
    """
    boots = [roc_auc_score(y[i], proba[i]) for i in _bootstrap_indices(y, n_boot, seed)]
    low, high = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"auc": float(roc_auc_score(y, proba)), "ic_inf": float(low), "ic_sup": float(high)}


def paired_bootstrap_auc_diff(
    y: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    n_boot: int = PROBE_BOOTSTRAP_N,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """Diferencia de AUC (A - B) sobre las mismas galaxias, con IC bootstrap.

    Comparación pareada: cada remuestreo toma las mismas galaxias para A y B,
    así la variación de "qué galaxias tocaron" se cancela y solo queda la
    diferencia entre representaciones.

    Args:
        y: etiquetas 0/1.
        proba_a: probabilidades del modelo/variante A.
        proba_b: probabilidades del modelo/variante B, mismas galaxias y orden.
        n_boot: número de remuestreos.
        alpha: 1 - nivel de confianza.
        seed: semilla del remuestreo.

    Returns:
        `dif_auc`, `ic_inf`, `ic_sup` y `p_valor` (bilateral: fracción de
        remuestreos en los que la diferencia cambia de signo, x2).
    """
    diffs = np.array([
        roc_auc_score(y[i], proba_a[i]) - roc_auc_score(y[i], proba_b[i])
        for i in _bootstrap_indices(y, n_boot, seed)
    ])
    low, high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    observed = roc_auc_score(y, proba_a) - roc_auc_score(y, proba_b)
    p_value = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"dif_auc": float(observed), "ic_inf": float(low), "ic_sup": float(high), "p_valor": float(p_value)}

def cross_group_auc(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> List[Dict[str, object]]:
    """Entrena con un grupo y evalúa con cada uno de los otros (p. ej. norte -> sur).

    Mide si lo aprendido en un dominio (telescopio, región del cielo) se
    transfiere al otro. Una caída fuerte respecto a la validación cruzada
    mezclada indica que el modelo depende del dominio y no solo de la
    morfología.

    Args:
        estimator: clasificador de sklearn con `predict_proba` (se clona).
        X: matriz de features.
        y: etiquetas 0/1.
        groups: grupo de cada muestra (p. ej. la región de `legacy_region`).

    Returns:
        Lista de diccionarios con `entrena`, `evalua`, `n_evalua` y `auc`
        (NaN si el grupo evaluado tiene una sola clase).
    """
    results = []
    for train_group in np.unique(groups):
        train = groups == train_group
        model = clone(estimator).fit(X[train], y[train])
        for test_group in np.unique(groups):
            if test_group == train_group:
                continue
            test = groups == test_group
            auc = (
                roc_auc_score(y[test], model.predict_proba(X[test])[:, 1])
                if len(np.unique(y[test])) == 2 else np.nan
            )
            results.append({"entrena": train_group, "evalua": test_group,
                            "n_evalua": int(test.sum()), "auc": float(auc)})
    return results


def multi_seed_paired_comparison(
    estimator: BaseEstimator,
    representations: Dict[str, np.ndarray],
    y: np.ndarray,
    reference: str,
    seeds: Sequence[int],
    n_splits: int = SHORTCUT_CV_FOLDS,
) -> List[Dict[str, object]]:
    """Compara varias representaciones de las mismas muestras contra una de referencia.

    Para cada semilla se obtienen predicciones *out-of-fold* con las mismas
    particiones para todas las representaciones. Se reporta el AUC medio entre
    semillas y, contra `reference`, la diferencia pareada
    (`paired_bootstrap_auc_diff`) sobre la probabilidad promediada entre
    semillas, que es menos sensible a cómo cayeron los folds que una sola
    partición. `semillas_mejor` cuenta en cuántas semillas la representación
    supera a la referencia.

    Args:
        estimator: clasificador de sklearn con `predict_proba` (se clona).
        representations: nombre -> matriz de features, mismas filas y orden.
        y: etiquetas 0/1.
        reference: nombre de la representación de referencia.
        seeds: semillas de las particiones.
        n_splits: folds de la validación cruzada.

    Returns:
        Una fila por representación con `representacion`, `auc_media`,
        `auc_std_semillas` y, salvo la referencia, `dif_auc`, `ic_inf`,
        `ic_sup`, `p_valor` y `semillas_mejor`.
    """
    oof = {
        key: np.stack([out_of_fold_proba(clone(estimator), X, y, n_splits, seed) for seed in seeds])
        for key, X in representations.items()
    }
    ref_aucs = np.array([roc_auc_score(y, p) for p in oof[reference]])
    rows = []
    for key, probas in oof.items():
        aucs = np.array([roc_auc_score(y, p) for p in probas])
        row: Dict[str, object] = {"representacion": key, "auc_media": float(aucs.mean()),
                                  "auc_std_semillas": float(aucs.std())}
        if key != reference:
            row.update(paired_bootstrap_auc_diff(y, probas.mean(axis=0), oof[reference].mean(axis=0)))
            row["semillas_mejor"] = int((aucs > ref_aucs).sum())
        rows.append(row)
    return rows

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
