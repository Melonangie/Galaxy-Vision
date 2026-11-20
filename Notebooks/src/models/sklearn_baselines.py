"""Modelos baseline de sklearn para elíptica vs. espiral.

Comparación de varios clasificadores clásicos sobre píxeles aplanados
(`src/data/features.py`), como piso de referencia antes de invertir en la
CNN/ResNet de la fase 1 (ver README.md, sección Arquitectura del modelo). Cada modelo comparte el mismo
pipeline de escalado + reducción de dimensionalidad (PCA) para que la
comparación entre modelos sea justa.
"""

from typing import Dict

from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import PCA_N_COMPONENTS, RANDOM_SEED


def _pipeline(estimator, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=PCA_N_COMPONENTS, random_state=seed)),
            ("clf", estimator),
        ]
    )


def get_baseline_models(seed: int = RANDOM_SEED) -> Dict[str, Pipeline]:
    """Diccionario `nombre -> Pipeline` con los baselines de sklearn a comparar.

    Todos usan `class_weight="balanced"` (ver README.md, sección Manejo de desbalance de clases) salvo KNN, que
    no lo soporta de forma nativa en sklearn.

    Args:
        seed: semilla para los componentes con aleatoriedad.

    Returns:
        Diccionario de pipelines (`StandardScaler` + `PCA` + clasificador)
        listos para `fit`/`predict`/`predict_proba`.
    """
    return {
        "logistic_regression": _pipeline(
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            ),
            seed,
        ),
        "knn": _pipeline(KNeighborsClassifier(n_neighbors=15, n_jobs=-1), seed),
        "random_forest": _pipeline(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced", # Automatic balancing of classes
                random_state=seed,
                n_jobs=-1,
            ),
            seed,
        ),
        "svm_linear": _pipeline(
            CalibratedClassifierCV(
                LinearSVC(
                    class_weight="balanced", # Automatic balancing of classes
                    random_state=seed, max_iter=5000
                ),
                cv=3,
            ),
            seed,
        ),
        "hist_gradient_boosting": _pipeline(
            HistGradientBoostingClassifier(
                class_weight="balanced", # Automatic balancing of classes
                random_state=seed
            ),
            seed,
        ),
    }

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
