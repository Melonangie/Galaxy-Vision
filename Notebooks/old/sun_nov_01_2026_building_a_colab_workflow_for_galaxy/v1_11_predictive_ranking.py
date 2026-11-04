from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score, ConfusionMatrixDisplay

model_data = df.dropna(subset=["ttype"]).copy()
counts = model_data["ttype"].value_counts()

# Do not silently merge or discard rare classes.
if len(counts) < 2 or counts.min() < 10 or len(available) == 0:
    print("Predictive ranking skipped: need at least two classes and ≥10 galaxies/class.")
else:
    X = model_data[available].replace([np.inf, -np.inf], np.nan)
    y = model_data["ttype"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED,
    )

    # Feature eligibility is decided using training data only.
    selected = [
        column for column in X_train
        if X_train[column].notna().mean() >= 0.7
        and X_train[column].nunique(dropna=True) > 1
    ]

    if not selected:
        print("No sufficiently complete, nonconstant training features.")
    else:
        X_train, X_test = X_train[selected], X_test[selected]
        folds = min(5, y_train.value_counts().min())
        cv = StratifiedKFold(folds, shuffle=True, random_state=RANDOM_SEED)

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        )

        # The imputer is fit separately within every training fold.
        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["balanced_accuracy", "f1_macro"],
        )
        print("Training CV balanced accuracy:", scores["test_balanced_accuracy"].mean())
        print("Training CV macro F1:", scores["test_f1_macro"].mean())

        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        print("Held-out baseline balanced accuracy:",
              balanced_accuracy_score(y_test, baseline.predict(X_test)))
        print("Held-out model balanced accuracy:",
              balanced_accuracy_score(y_test, prediction))
        print("Held-out model macro F1:",
              f1_score(y_test, prediction, average="macro"))

        importance = permutation_importance(
            model, X_test, y_test,
            scoring="balanced_accuracy",
            n_repeats=20,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        importance_table = pd.DataFrame({
            "feature": selected,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }).sort_values("importance_mean", ascending=False)

        importance_table.to_csv(
            OUTPUT_DIR / "heldout_permutation_importance.csv", index=False
        )
        display(importance_table)

        ConfusionMatrixDisplay.from_predictions(
            y_test, prediction, normalize="true", xticks_rotation=45,
        )
        plt.title("Held-out morphology classification")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "heldout_confusion_matrix.png", dpi=170)
        plt.show()