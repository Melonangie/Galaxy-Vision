"""Derivación de la etiqueta binaria (elíptica vs. espiral) a partir de las
vote fractions de Galaxy Zoo 2.

No existe una columna de clase discreta en el dataset original: hay que
derivarla de Class1.1 (smooth -> proxy de elíptica) y Class1.2
(features/disk -> proxy de espiral), siguiendo el criterio de vote fraction
"confiable" (>= 0.8) usado por Walmsley et al. (2022) y consistente con el
subconjunto de alto acuerdo de Dieleman, Willett & Dambre (2015).
Ver README.md, sección Dataset.
"""

import pandas as pd

from src.config import (
    AMBIGUOUS_LABEL,
    CLEAN_LABEL_THRESHOLD,
    ELLIPTICAL_LABEL,
    SPIRAL_LABEL,
)


def derive_binary_label(
    df: pd.DataFrame, threshold: float = CLEAN_LABEL_THRESHOLD
) -> pd.DataFrame:
    """Añade una columna `label` con la etiqueta binaria elíptica/espiral.

    Reglas:
        elíptica  si Class1.1 >= threshold
        espiral   si Class1.2 >= threshold
        ambigua   en cualquier otro caso (bajo consenso o artefacto)

    Args:
        df: DataFrame con al menos las columnas `Class1.1` y `Class1.2`
            (formato de `training_solutions_rev1.csv`).
        threshold: umbral de vote fraction para considerar la clase confiable.

    Returns:
        Copia de `df` con una columna adicional `label` (`"elliptical"`,
        `"spiral"` o `"ambigua"`).
    """
    out = df.copy()
    out["label"] = AMBIGUOUS_LABEL
    out.loc[out["Class1.1"] >= threshold, "label"] = ELLIPTICAL_LABEL
    out.loc[out["Class1.2"] >= threshold, "label"] = SPIRAL_LABEL
    return out

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
