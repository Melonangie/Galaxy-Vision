overall_rows = []
pairwise_rows = []

for column in candidate_columns:
    groups = {
        t: group[column].dropna().to_numpy()
        for t, group in features.groupby("ttype")
    }
    groups = {t: values for t, values in groups.items() if len(values) >= 10}
    if len(groups) < 2:
        continue

    pooled = np.concatenate(list(groups.values()))
    if np.unique(pooled).size < 2:
        continue

    statistic, p_value = kruskal(*groups.values())
    n, k = len(pooled), len(groups)

    # Rank-based overall effect-size approximation.
    epsilon_squared = max(0, (statistic - k + 1) / (n - k))
    overall_rows.append({
        "feature": column, "n": n, "types": k,
        "H": statistic, "p": p_value,
        "epsilon_squared": epsilon_squared,
    })

    for first, second in combinations(groups, 2):
        a, b = groups[first], groups[second]
        test = mannwhitneyu(a, b, alternative="two-sided")

        # Signed rank-biserial effect size; magnitude near 0 = overlap.
        effect = 2 * test.statistic / (len(a) * len(b)) - 1
        pairwise_rows.append({
            "feature": column, "ttype_a": first, "ttype_b": second,
            "n_a": len(a), "n_b": len(b),
            "rank_biserial": effect, "absolute_effect": abs(effect),
            "p": test.pvalue,
        })

overall = pd.DataFrame(overall_rows)
pairwise = pd.DataFrame(pairwise_rows)

if not overall.empty:
    overall["q_fdr"] = false_discovery_control(overall["p"], method="bh")
    overall = overall.sort_values("epsilon_squared", ascending=False)
    overall.to_csv(OUTPUT_DIR / "overall_feature_separation.csv", index=False)
    display(overall)

if not pairwise.empty:
    pairwise["q_fdr"] = false_discovery_control(pairwise["p"], method="bh")
    pairwise = pairwise.sort_values("absolute_effect", ascending=False)
    pairwise.to_csv(OUTPUT_DIR / "pairwise_feature_separation.csv", index=False)
    display(pairwise.head(30))