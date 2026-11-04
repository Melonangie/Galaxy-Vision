from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay

rank = []
for f in FEATURES:
    groups = [g[f].dropna().values for _, g in d.groupby("Ttype") if g[f].notna().sum() >= 5]
    if len(groups) < 2: continue
    H, p = kruskal(*groups)
    rho, _ = spearmanr(d[f], d.Ttype, nan_policy="omit")
    rank.append({"feature": f, "epsilon2": H / (sum(map(len, groups)) - 1), "p_value": p,
                 "spearman_Ttype": rho, "coverage": d[f].notna().mean()})
rank = pd.DataFrame(rank).set_index("feature")
feats_ok = list(rank.index)

# Keep T-types with at least 10 galaxies for the classifier
keep = d.Ttype.map(d.Ttype.value_counts()) >= 10
X, y = d.loc[keep, feats_ok], d.loc[keep, "Ttype"]
rank["mutual_info"] = mutual_info_classif(SimpleImputer(strategy="median").fit_transform(X), y, random_state=0)

model = make_pipeline(SimpleImputer(strategy="median"),
                      RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                             random_state=0, n_jobs=-1))
print("5-fold balanced accuracy:", cross_val_score(model, X, y, cv=5, scoring="balanced_accuracy").mean().round(3))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
model.fit(Xtr, ytr)
rank["perm_importance"] = permutation_importance(model, Xte, yte, n_repeats=10,
                                                 scoring="balanced_accuracy", random_state=0).importances_mean
rank = rank.sort_values("epsilon2", ascending=False)
rank.to_csv(f"{SAVE_DIR}/feature_ranking.csv"); display(rank.round(3))

rank[["epsilon2", "mutual_info", "perm_importance"]].plot.barh(subplots=True, layout=(1, 3),
                                                               figsize=(18, 10), sharey=True, legend=False)
plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()

# Confusion matrix: mistakes are usually between neighbouring T-types
ConfusionMatrixDisplay.from_estimator(model, Xte, yte, normalize="true", cmap="Blues", values_format=".2f")
plt.title("Normalised confusion matrix"); plt.show()