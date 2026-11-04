EMI_FEATS = [f"EW_{k}" for k in EMISSION]                         # all emission EWs
ABS_FEATS = ["CaII_K", "CaII_H", "G4300", "Mgb", "NaD", "CaT", "Dn4000", "color_4500_6500"]
POP_FEATS = ["logage_L", "logage_M", "logZ_L", "logZ_M", "xY", "xI", "xO", "sl_AV", "sl_vd"]
RATIOS    = ["log_NII_Ha", "log_OIII_Hb", "log_SII_Ha", "log_OI_Ha", "log_NeIII_OII", "Ha_Hb", "AV_gas"]
FEATURES  = POP_FEATS + ABS_FEATS + EMI_FEATS + RATIOS + ["LogMass"]

d = feat[feat.good_fit]

def boxgrid(cols, fname, ncol=4):
    n = len(cols)
    fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(22, 3.3 * np.ceil(n / ncol)))
    for ax, f in zip(axes.flat, cols):
        sns.boxplot(data=d, x="Ttype", y=f, palette=TT_COLORS, ax=ax, showfliers=False)
        ax.set_xlabel("")
    for ax in axes.flat[n:]: ax.remove()
    plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/{fname}", dpi=110); plt.show()

boxgrid(POP_FEATS, "box_populations.png")
boxgrid(ABS_FEATS, "box_absorption.png")
boxgrid(EMI_FEATS, "box_emission.png")
boxgrid(RATIOS,    "box_ratios.png")

# Feature-feature correlation (pairs with very high |rho| carry the same information)
plt.figure(figsize=(16, 14))
sns.heatmap(d[FEATURES + ["Ttype"]].corr(method="spearman"), cmap="coolwarm", center=0)
plt.title("Spearman correlation"); plt.show()