x = np.linspace(-2, 1, 300)
kauff = lambda x: 0.61 / (x - 0.05) + 1.30          # Kauffmann+03
kew01 = lambda x: 0.61 / (x - 0.47) + 1.19          # Kewley+01

def bpt_class(nii, oiii):
    if np.isnan(nii) or np.isnan(oiii): return "No/weak lines"
    if nii < 0.05 and oiii < kauff(nii): return "Star-forming"
    if nii < 0.47 and oiii < kew01(nii): return "Composite"
    return "Seyfert" if oiii > 1.05 * nii + 0.45 else "LINER"   # Schawinski+07

def whan_class(nii, ew_ha, ew_nii):
    if ew_ha < 0.5 and ew_nii < 0.5: return "Passive"
    if ew_ha < 3:                    return "Retired"
    if np.isnan(nii):                return "Unclassified"
    if nii < -0.4:                   return "SF"
    return "sAGN" if ew_ha > 6 else "wAGN"

feat["BPT"]  = [bpt_class(a, b) for a, b in zip(feat.log_NII_Ha, feat.log_OIII_Hb)]
feat["WHAN"] = [whan_class(a, b, c) for a, b, c in zip(feat.log_NII_Ha, feat.EW_Ha, feat.EW_NII6584)]
d = feat[feat.good_fit]

# --- [NII] BPT, one panel per class
fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
for ax, c in zip(axes.flat, CLASS_ORDER):
    g = d[d.morph_class == c]
    ax.scatter(g.log_NII_Ha, g.log_OIII_Hb, s=8, alpha=0.6, color=CLASS_COLORS[c])
    ax.plot(x[x < 0.05], kauff(x[x < 0.05]), "k--", lw=1, label="Kauffmann+03")
    ax.plot(x[x < 0.47], kew01(x[x < 0.47]), "k-",  lw=1, label="Kewley+01")
    ax.plot(np.linspace(-0.18, 1, 50), 1.05 * np.linspace(-0.18, 1, 50) + 0.45, "k:", lw=1, label="Schawinski+07")
    ax.set_title(f"{c}  (N with lines = {g.log_OIII_Hb.notna().sum()}/{len(g)})")
    ax.set_xlim(-1.8, 0.8); ax.set_ylim(-1.5, 1.5)
axes[0, 0].legend(fontsize=8)
fig.supxlabel("log [NII]6584/Hα"); fig.supylabel("log [OIII]5007/Hβ")
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/BPT_NII_by_class.png", dpi=120); plt.show()

# --- [SII] and [OI] BPT (Kewley+06 lines), all classes coloured
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
for c in CLASS_ORDER:
    g = d[d.morph_class == c]
    ax[0].scatter(g.log_SII_Ha, g.log_OIII_Hb, s=6, alpha=0.5, color=CLASS_COLORS[c], label=c)
    ax[1].scatter(g.log_OI_Ha,  g.log_OIII_Hb, s=6, alpha=0.5, color=CLASS_COLORS[c])
xs = np.linspace(-1.5, 0.3, 100); ax[0].plot(xs, 0.72 / (xs - 0.32) + 1.30, "k-")
ax[0].plot(np.linspace(-0.31, 0.5), 1.89 * np.linspace(-0.31, 0.5) + 0.76, "k:")
xo = np.linspace(-2.5, -0.6, 100); ax[1].plot(xo, 0.73 / (xo + 0.59) + 1.33, "k-")
ax[1].plot(np.linspace(-1.1, 0.5), 1.18 * np.linspace(-1.1, 0.5) + 1.30, "k:")
ax[0].set_xlabel("log [SII]/Hα"); ax[1].set_xlabel("log [OI]/Hα"); ax[0].set_ylabel("log [OIII]/Hβ")
for a in ax: a.set_ylim(-1.5, 1.5)
ax[0].legend(markerscale=3); plt.show()

# --- WHAN diagram
plt.figure(figsize=(8, 6))
for c in CLASS_ORDER:
    g = d[d.morph_class == c]
    plt.scatter(g.log_NII_Ha, g.log_EW_Ha, s=6, alpha=0.5, color=CLASS_COLORS[c], label=c)
plt.axhline(np.log10(3), c="k"); plt.axhline(np.log10(6), c="k", ls=":", xmin=0.5); plt.axvline(-0.4, c="k")
plt.xlabel("log [NII]/Hα"); plt.ylabel("log EW(Hα) [Å]"); plt.legend(markerscale=3); plt.title("WHAN"); plt.show()

# --- How the BPT/WHAN classes are spread within each morphology (main result table)
for col in ["BPT", "WHAN"]:
    ct = pd.crosstab(d.morph_class, d[col], normalize="index").reindex(CLASS_ORDER)
    display((100 * ct).round(1))
    ct.plot.bar(stacked=True, figsize=(10, 4), colormap="tab10"); plt.title(f"{col} class fraction per morphology")
    plt.legend(bbox_to_anchor=(1, 1)); plt.show()

# --- Balmer decrement, dust and age-sensitive absorption per class
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(data=d, x="morph_class", y="Ha_Hb", order=CLASS_ORDER, palette=CLASS_COLORS, ax=ax[0], showfliers=False)
ax[0].axhline(2.86, c="k", ls="--", label="Case B (2.86)"); ax[0].legend()
sns.boxplot(data=d, x="morph_class", y="AV_gas", order=CLASS_ORDER, palette=CLASS_COLORS, ax=ax[1], showfliers=False)
for c in CLASS_ORDER:                               # Dn4000-HδA plane (Kauffmann et al. 2003a)
    g = d[d.morph_class == c]
    ax[2].scatter(g.Dn4000, g.HdA, s=5, alpha=0.5, color=CLASS_COLORS[c], label=c)
ax[2].set_xlabel("Dn4000"); ax[2].set_ylabel("HδA [Å]"); ax[2].legend(markerscale=3)
plt.tight_layout(); plt.show()

# --- Summary table of line diagnostics per class (medians)
diag = ["EW_Ha", "EW_Hb", "EW_OII3727", "EW_OIII5007", "Ha_Hb", "AV_gas", "SII_ratio",
        "Dn4000", "HdA", "HgA", "Hb_abs", "Mgb", "Fe5270", "NaD"]
display(d.groupby("morph_class")[diag].median().reindex(CLASS_ORDER).round(2))
feat.to_csv(f"{SAVE_DIR}/features_with_diagnostics.csv")