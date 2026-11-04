from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay
)

data = features.dropna(subset=["ttype"]).copy()

# Sparse classes cannot support useful stratified evaluation.
counts = data["ttype"].value_counts()
eligible = counts[counts >= 5].index
excluded = counts[counts < 5]
print("Classes excluded from this baseline due to small counts:")
display(excluded)

data = data[data["ttype"].isin(eligible)].reset_index(drop=True)
usable_features = [
    c for c in candidate_columns if data[c].notna().sum() >= 10
]

if data["ttype"].nunique() < 2 or not usable_features:
    print("Not enough classes/features for classification.")
else:
    X = data[usable_features].replace([np.inf, -np.inf], np.nan)
    y = data["ttype"].astype(str)
    folds = min(5, int(y.value_counts().min()))

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    predictions = np.empty(len(y), dtype=object)
    importance_rows = []
    scores = []

    for fold, (train, test) in enumerate(cv.split(X, y), start=1):
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(
                n_estimators=200, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
        )
        model.fit(X.iloc[train], y.iloc[train])
        predictions[test] = model.predict(X.iloc[test])

        scores.append({
            "fold": fold,
            "balanced_accuracy": balanced_accuracy_score(
                y.iloc[test], predictions[test]
            ),
            "macro_f1": f1_score(
                y.iloc[test], predictions[test], average="macro"
            ),
        })

        importance = permutation_importance(
            model, X.iloc[test], y.iloc[test],
            scoring="f1_macro", n_repeats=10,
            random_state=42, n_jobs=-1
        )
        for column, mean, std in zip(
            usable_features, importance.importances_mean,
            importance.importances_std
        ):
            importance_rows.append({
                "fold": fold, "feature": column,
                "importance": mean, "repeat_std": std
            })

    score_table = pd.DataFrame(scores)
    importance_table = pd.DataFrame(importance_rows)
    ranking = (
        importance_table.groupby("feature")["importance"]
        .agg(["mean", "std"]).sort_values("mean", ascending=False)
    )

    score_table.to_csv(OUTPUT_DIR / "classification_cv_scores.csv", index=False)
    importance_table.to_csv(OUTPUT_DIR / "permutation_importance_by_fold.csv", index=False)
    ranking.to_csv(OUTPUT_DIR / "classification_feature_ranking.csv")

    display(score_table)
    display(ranking)

    labels = sorted(y.unique())
    matrix = confusion_matrix(y, predictions, labels=labels)
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        xticks_rotation=45, cmap="Blues"
    )
    plt.title("Out-of-fold morphology predictions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "classification_confusion_matrix.png", dpi=160)
    plt.show()