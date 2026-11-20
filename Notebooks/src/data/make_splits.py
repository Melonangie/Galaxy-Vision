"""Genera y guarda a CSV los splits train/val/test (subconjunto limpio).

Persiste los splits en `data/processed/` para que notebooks y scripts de
entrenamiento los consuman directamente sin rederivarlos (ver README.md,
sección Estructura del proyecto).

Uso:
    python -m src.data.make_splits
"""

from src.config import PROCESSED_DATA_DIR
from src.data.dataset import load_clean_dataset, save_splits_to_csv, stratified_split


def main() -> None:
    df_clean = load_clean_dataset()
    train_df, val_df, test_df = stratified_split(df_clean)
    save_splits_to_csv(train_df, val_df, test_df)

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        props = split["label"].value_counts(normalize=True)
        print(
            f"{name:6s} n={len(split):6d}  "
            f"elliptical={props.get('elliptical', 0):.1%}  "
            f"spiral={props.get('spiral', 0):.1%}"
        )
    print(f"\nCSVs guardados en {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
