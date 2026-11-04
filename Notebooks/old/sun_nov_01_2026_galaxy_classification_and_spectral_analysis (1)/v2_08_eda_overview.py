# 1) Galaxies per T-type: some T-types will be rare (Sdm/Irr in MaNGA) -> statistics are noisier
ax = feat.Ttype.value_counts().sort_index().plot.bar(color=[TT_COLORS[t] for t in sorted(feat.Ttype.unique())],
                                                       figsize=(10, 4))
ax.set_xticklabels([f"{t}\n{TT_NAMES.get(t,'')}" for t in sorted(feat.Ttype.unique())], rotation=0)
plt.title("Galaxies per T-type"); plt.savefig(f"{SAVE_DIR}/counts.png", dpi=120); plt.show()

# 2) Missing values per feature (emission lines are often missing in E/S0 because they are weak)
print(feat.isna().mean().sort_values(ascending=False).head(25))

# 3) STARLIGHT fit quality
print(feat[["sl_chi2", "sl_adev", "sl_SN"]].describe())
print("good fits:", feat.good_fit.mean().round(3))

# 4) Catalogue properties per T-type.
#    LogMass: early types are more massive (Kauffmann et al. 2003b).
#    incl: edge-on disks look redder/dustier -> keep this in mind for late types.
#    nsa_z, PETRO_TH90: different T-types may be observed at different distances and sizes.
fig, axes = plt.subplots(1, 4, figsize=(22, 4))
for ax, c in zip(axes, ["nsa_z", "LogMass", "incl", "PETRO_TH90"]):
    sns.boxplot(data=feat, x="Ttype", y=c, palette=TT_COLORS, ax=ax, showfliers=False)
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/catalog_props.png", dpi=120); plt.show()

# 5) Bar fraction per T-type (bars are most common in early/intermediate spirals; Masters et al. 2011)
pd.crosstab(feat.Ttype, feat.bar, normalize="index").plot.bar(stacked=True, figsize=(10, 4))
plt.title("Bar fraction per T-type"); plt.show()