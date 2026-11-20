"""Entrena y compara varios clasificadores de sklearn (elíptica vs. espiral).

Baseline clásico previo a la CNN/ResNet de la fase 1 (ver README.md, sección
Arquitectura del modelo): fija un piso de referencia y valida de punta a punta el pipeline de datos
(labeling -> split -> features) antes de invertir en deep learning.

Uso:
    python -m src.train_baselines
"""

import random
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    BASELINE_TRAIN_SAMPLE_SIZE,
    BASELINE_VAL_SAMPLE_SIZE,
    FEATURE_IMAGE_SIZE,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.data.dataset import load_clean_dataset, stratified_split
from src.data.features import extract_pixel_features
from src.evaluate import evaluate_binary_classifier
from src.models.sklearn_baselines import get_baseline_models


def _stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Submuestra `df` a lo sumo `n` filas, preservando la proporción de clases."""
    if len(df) <= n:
        return df
    subsample, _ = train_test_split(
        df, train_size=n, stratify=df["label"], random_state=seed
    )
    return subsample.reset_index(drop=True)


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Cargando subconjunto limpio (elíptica/espiral)...")
    df_clean = load_clean_dataset()
    train_df, val_df, _test_df = stratified_split(df_clean)

    train_sample = _stratified_subsample(train_df, BASELINE_TRAIN_SAMPLE_SIZE, RANDOM_SEED)
    val_sample = _stratified_subsample(val_df, BASELINE_VAL_SAMPLE_SIZE, RANDOM_SEED)
    print(
        f"Train: {len(train_sample)}/{len(train_df)} usadas | "
        f"Val: {len(val_sample)}/{len(val_df)} usadas "
        "(submuestreo estratificado para mantener tiempos de cómputo razonables "
        "con KNN/SVM; ver BASELINE_*_SAMPLE_SIZE en src/config.py)"
    )

    print(f"Extrayendo features de píxeles ({FEATURE_IMAGE_SIZE}x{FEATURE_IMAGE_SIZE})...")
    X_train, y_train = extract_pixel_features(train_sample)
    X_val, y_val = extract_pixel_features(val_sample)

    models = get_baseline_models(seed=RANDOM_SEED)
    results = []

    for name, pipeline in models.items():
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        elapsed = time.time() - t0

        metrics = evaluate_binary_classifier(y_val, y_pred, y_proba)
        report = metrics["report"]
        results.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "precision_spiral": report["1"]["precision"],
                "recall_spiral": report["1"]["recall"],
                "f1_spiral": report["1"]["f1-score"],
                "precision_elliptical": report["0"]["precision"],
                "recall_elliptical": report["0"]["recall"],
                "f1_elliptical": report["0"]["f1-score"],
                "train_seconds": elapsed,
            }
        )
        print(f"\n=== {name} ({elapsed:.1f}s) ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}  ROC-AUC: {metrics['roc_auc']:.4f}")
        print("Matriz de confusión (filas=real, cols=predicho; orden [elliptical, spiral]):")
        print(metrics["confusion_matrix"])

    summary = (
        pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    )
    print("\n=== Comparación de modelos (ordenado por ROC-AUC) ===")
    print(summary.to_string(index=False))

    best = summary.iloc[0]
    print(
        f"\nMejor baseline: {best['model']} "
        f"(accuracy={best['accuracy']:.4f}, ROC-AUC={best['roc_auc']:.4f})"
    )
    print(
        "Referencia de literatura (ver README.md, sección Métricas de evaluación): "
        "~94.5-99% OA en binario "
        "con CNN/transfer learning. Un baseline clásico sobre píxeles planos "
        "debería, por diseño, quedar por debajo de ese rango: confirma que el "
        "siguiente paso es la CNN/ResNet de la fase 1, no que el pipeline esté mal."
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[best["model"]], MODELS_DIR / "best_sklearn_baseline.joblib")
    summary.to_csv(MODELS_DIR / "sklearn_baselines_comparison.csv", index=False)
    print(f"\nModelo y comparación guardados en {MODELS_DIR}")


if __name__ == "__main__":
    main()

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
