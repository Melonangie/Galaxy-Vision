from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance

rank = []
for f in FEATURES:
    groups = [g[f].dropna().values for _, g in d.groupby("morph_class") if g[f].notna().sum() > 5]
    H, p = kruskal(*groups)
    n_obs = sum(len(g) for g in groups)
    rho, _ = spearmanr(d[f], d["Ttype"], nan_policy="omit")
    rank.append({"feature": f, "epsilon2": H / (n_obs - 1), "p_value": p, "spearman_Ttype": rho,
                 "coverage": d[f].notna().mean()})
rank = pd.DataFrame(rank).set_index("feature")

# Machine-learning ranking (missing values -> median; balanced classes)
X = d[FEATURES]; y = d["morph_class"]
Xi = SimpleImputer(strategy="median").fit_transform(X)
rank["mutual_info"] = mutual_info_classif(Xi, y, random_state=0)

model = make_pipeline(SimpleImputer(strategy="median"),
                      RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=0, n_jobs=-1))
print("5-fold balanced accuracy:", cross_val_score(model, X, y, cv=5, scoring="balanced_accuracy").mean().round(3))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
model.fit(Xtr, ytr)
pi = permutation_importance(model, Xte, yte, n_repeats=10, scoring="balanced_accuracy", random_state=0)
rank["perm_importance"] = pi.importances_mean

rank = rank.sort_values("epsilon2", ascending=False)
rank.to_csv(f"{SAVE_DIR}/feature_ranking.csv")
display(rank.round(3))

rank[["epsilon2", "mutual_info", "perm_importance"]].plot.barh(subplots=True, layout=(1, 3), figsize=(18, 8),
                                                               sharey=True, legend=False)
plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()