FEATURES = [
    # stellar populations (STARLIGHT)
    "logage_L", "logage_M", "logZ_L", "logZ_M", "xY", "xI", "xO", "sl_AV", "sl_vd",
    # absorption features
    "Dn4000", "HdA", "HgA", "Hb_abs", "Mgb", "Fe5270", "Fe5335", "NaD", "color_4500_6500",
    # emission features
    "log_EW_Ha", "EW_OII3727", "EW_Hb", "log_NII_Ha", "log_OIII_Hb", "log_SII_Ha", "Ha_Hb", "AV_gas",
    # catalogue
    "LogMass",
]
d = feat[feat.good_fit]

n = len(FEATURES); ncol = 4
fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(20, 3.2 * np.ceil(n / ncol)))
for ax, f in zip(axes.flat, FEATURES):
    sns.boxplot(data=d, x="morph_class", y=f, order=CLASS_ORDER, palette=CLASS_COLORS, ax=ax, showfliers=False)
    ax.set_xlabel(""); ax.tick_params(axis="x", rotation=45)
for ax in axes.flat[n:]: ax.remove()
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/feature_boxplots.png", dpi=110); plt.show()

# Correlations between features: strongly correlated features carry the same information
plt.figure(figsize=(14, 12))
sns.heatmap(d[FEATURES + ["Ttype"]].corr(method="spearman"), cmap="coolwarm", center=0, annot=False)
plt.title("Spearman correlation"); plt.show()