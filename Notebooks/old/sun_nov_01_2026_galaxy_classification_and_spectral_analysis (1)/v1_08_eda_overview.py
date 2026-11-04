# 1) How many galaxies per class? (class imbalance changes how classifiers must be trained)
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
feat["morph_class"].value_counts().reindex(CLASS_ORDER).plot.bar(ax=ax[0], color=[CLASS_COLORS[c] for c in CLASS_ORDER])
ax[0].set_title("Galaxies per morphological class")
feat["Ttype"].value_counts().sort_index().plot.bar(ax=ax[1], color="gray"); ax[1].set_title("Galaxies per T-type")
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/counts.png", dpi=120); plt.show()

# 2) Missing values: which features exist for which galaxies
miss = feat.isna().mean().sort_values(ascending=False)
print("Fraction missing (top 20):\n", miss.head(20))

# 3) STARLIGHT fit quality: discard bad fits from the analysis
print(feat[["sl_chi2", "sl_adev", "sl_SN"]].describe())
feat["good_fit"] = (feat.sl_adev < 10) & (feat.sl_SN > 5)   # adjust if needed

# 4) Catalogue properties by class.
# Mass rises toward early types (Roberts & Haynes 1994). Redshift and size show selection effects.
# Inclination matters: edge-on disks are redder and dustier, so spectra of late types change with inclination.
cat_cols = ["nsa_z", "LogMass", "incl", "PETRO_TH90"]
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, c in zip(axes, cat_cols):
    sns.boxplot(data=feat, x="morph_class", y=c, order=CLASS_ORDER, palette=CLASS_COLORS, ax=ax, showfliers=False)
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/catalog_props.png", dpi=120); plt.show()

# 5) Bars by class (barred fraction peaks in early/intermediate spirals; e.g. Masters et al. 2011)
pd.crosstab(feat.morph_class, feat.bar, normalize="index").reindex(CLASS_ORDER).plot.bar(stacked=True, figsize=(8, 4))
plt.title("Bar fraction per class"); plt.show()