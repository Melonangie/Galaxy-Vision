CANDIDATES = [
    "dn4000", "continuum_red_blue",
    "hdelta_a_observed_ew", "hgamma_a_observed_ew",
    "mgb_observed_ew", "fe5270_observed_ew", "fe5335_observed_ew",
    "light_mean_log_age", "mass_mean_log_age",
    "light_mean_z", "mass_mean_z",
    "light_young_fraction", "light_intermediate_fraction",
    "stellar_av", "stellar_vd_observed",
    "ha_ew_emission", "bpt_x", "bpt_y", "nebular_ebv_approx",
]
available = [
    name for name in CANDIDATES
    if name in df and pd.to_numeric(df[name], errors="coerce").notna().sum() >= 5
]

if available:
    missingness = df.groupby("ttype")[available].agg(
        lambda column: column.isna().mean()
    )
    missingness.to_csv(OUTPUT_DIR / "feature_missing_fraction_by_ttype.csv")

    for name in available:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=df, x="ttype", y=name, ax=ax, showfliers=False)
        sns.stripplot(data=df, x="ttype", y=name, ax=ax, size=2, alpha=0.3, color="black")
        ax.set_title(f"{name}: compare distributions, not only class means")
        fig.savefig(OUTPUT_DIR / f"feature_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

# Exploratory univariate ranking. Report unequal/missing sample sizes.
ranking = []
for name in available:
    groups = [
        group[name].dropna().to_numpy()
        for _, group in df.groupby("ttype")
    ]
    groups = [group for group in groups if len(group) >= 5]
    if len(groups) < 2:
        continue

    try:
        statistic, pvalue = kruskal(*groups)
    except ValueError:
        continue

    n = sum(map(len, groups))
    k = len(groups)
    effect = max(0, (statistic - k + 1) / (n - k))
    ranking.append({
        "feature": name, "kruskal_h": statistic, "p": pvalue,
        "epsilon_squared_approx": effect, "n": n, "classes": k,
    })

ranking = pd.DataFrame(ranking)
if not ranking.empty:
    ranking = ranking.sort_values("p").reset_index(drop=True)

    # Benjamini–Hochberg correction for multiple feature comparisons.
    adjusted = ranking["p"].to_numpy() * len(ranking) / np.arange(1, len(ranking) + 1)
    ranking["q_bh"] = np.minimum(
        1, np.minimum.accumulate(adjusted[::-1])[::-1]
    )
    ranking = ranking.sort_values("epsilon_squared_approx", ascending=False)
    ranking.to_csv(OUTPUT_DIR / "univariate_feature_ranking.csv", index=False)
    display(ranking)
else:
    print("Not enough galaxies per class for univariate separation analysis.")