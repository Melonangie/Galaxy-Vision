candidate_columns = [
    "Dn4000", "continuum_red_blue",
    "light_log_age", "mass_log_age",
    "young_light_fraction", "old_light_fraction",
    "light_Z", "mass_Z", "stellar_av",
    "HdeltaA_observed_ew", "HdeltaA_stellar_model_ew",
    "CaK_observed_ew", "Gband_observed_ew",
    "Mgb_observed_ew", "NaD_observed_ew",
    "CaII_triplet_observed_ew",
    "emission_ew_Halpha", "emission_ew_Hbeta",
    "Halpha_Hbeta", "gas_ebv_signed",
    "bpt_NII_x", "bpt_SII_x", "bpt_OI_x", "bpt_y",
    "O32_observed", "NeIII_OII_observed",
]
candidate_columns = [c for c in candidate_columns if c in features]

summary = features.groupby("ttype")[candidate_columns].agg(
    ["count", "median", lambda values: values.quantile(0.75) - values.quantile(0.25)]
)
summary.to_csv(OUTPUT_DIR / "feature_summary_by_ttype.csv")

available_fraction = features.groupby("ttype")[candidate_columns].agg(
    lambda values: values.notna().mean()
)
available_fraction.to_csv(OUTPUT_DIR / "feature_available_fraction_by_ttype.csv")

for column in candidate_columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=features, x="ttype", y=column, showfliers=False, ax=ax)
    ax.set_title(f"{column} by catalog Ttype")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / f"distribution_{column}.png", dpi=150)
    plt.close(fig)

display(summary)