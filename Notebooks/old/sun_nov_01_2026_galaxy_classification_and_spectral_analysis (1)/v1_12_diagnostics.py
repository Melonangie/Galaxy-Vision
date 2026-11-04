kauff = lambda x: 0.61 / (x - 0.05) + 1.30       # Kauffmann+03
kew01 = lambda x: 0.61 / (x - 0.47) + 1.19       # Kewley+01

def bpt_class(n2, o3):
    if np.isnan(n2) or np.isnan(o3): return "No/weak lines"
    if n2 < 0.05 and o3 < kauff(n2):  return "Star-forming"
    if n2 < 0.47 and o3 < kew01(n2):  return "Composite"
    return "Seyfert" if o3 > 1.05 * n2 + 0.45 else "LINER"

def whan_class(n2, ew_ha, ew_n2):
    if ew_ha < 0.5 and ew_n2 < 0.5: return "Passive"
    if ew_ha < 3:                   return "Retired"
    if np.isnan(n2):                return "Unclassified"
    if n2 < -0.4:                   return "SF"
    return "sAGN" if ew_ha > 6 else "wAGN"

feat["BPT"]  = [bpt_class(a, b) for a, b in zip(feat.log_NII_Ha, feat.log_OIII_Hb)]
feat["WHAN"] = [whan_class(a, b, c) for a, b, c in zip(feat.log_NII_Ha, feat.EW_Ha, feat.EW_NII6584)]
d = feat[feat.good_fit]
x = np.linspace(-2, 1, 300)

# ---- [NII] BPT: one panel per T-type
tts = sorted(d.Ttype.unique()); ncol = 4
fig, axes = plt.subplots(int(np.ceil(len(tts) / ncol)), ncol, figsize=(20, 4.5 * np.ceil(len(tts) / ncol)),
                         sharex=True, sharey=True)
for ax, t in zip(axes.flat, tts):
    g = d[d.Ttype == t]
    ax.scatter(g.log_NII_Ha, g.log_OIII_Hb, s=8, alpha=0.6, color=TT_COLORS[t])
    ax.plot(x[x < 0.05], kauff(x[x < 0.05]), "k--", lw=1)
    ax.plot(x[x < 0.47], kew01(x[x < 0.47]), "k-", lw=1)
    xs = np.linspace(-0.18, 1, 50); ax.plot(xs, 1.05 * xs + 0.45, "k:", lw=1)
    ax.set_title(f"T={t} {TT_NAMES.get(t,'')}  ({g.log_OIII_Hb.notna().sum()}/{len(g)} with lines)")
    ax.set_xlim(-1.8, 0.8); ax.set_ylim(-1.5, 1.5)
for ax in axes.flat[len(tts):]: ax.remove()
fig.supxlabel("log [NII]6584/Hα"); fig.supylabel("log [OIII]5007/Hβ")
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/BPT_NII_by_ttype.png", dpi=110); plt.show()

# ---- [SII] and [OI] BPT, coloured by T-type (Kewley+06 lines)
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
for a, xcol in zip(ax, ["log_SII_Ha", "log_OI_Ha"]):
    sc = a.scatter(d[xcol], d.log_OIII_Hb, c=d.Ttype, cmap="RdYlBu", s=6, alpha=0.6)
    a.set_xlabel(xcol); a.set_ylim(-1.5, 1.5)
xs = np.linspace(-1.5, 0.3, 100); ax[0].plot(xs, 0.72 / (xs - 0.32) + 1.30, "k-")
xs = np.linspace(-0.31, 0.5); ax[0].plot(xs, 1.89 * xs + 0.76, "k:")
xo = np.linspace(-2.5, -0.6, 100); ax[1].plot(xo, 0.73 / (xo + 0.59) + 1.33, "k-")
xo = np.linspace(-1.1, 0.5); ax[1].plot(xo, 1.18 * xo + 1.30, "k:")
ax[0].set_ylabel("log [OIII]/Hβ"); plt.colorbar(sc, ax=ax, label="T-type"); plt.show()

# ---- WHAN
plt.figure(figsize=(8, 6))
sc = plt.scatter(d.log_NII_Ha, d.log_EW_Ha, c=d.Ttype, cmap="RdYlBu", s=6, alpha=0.6)
plt.axhline(np.log10(3), c="k"); plt.axhline(np.log10(6), c="k", ls=":"); plt.axvline(-0.4, c="k")
plt.colorbar(sc, label="T-type"); plt.xlabel("log [NII]/Hα"); plt.ylabel("log EW(Hα) [Å]"); plt.title("WHAN"); plt.show()

# ---- Fraction of BPT / WHAN classes in each T-type
for col in ["BPT", "WHAN"]:
    ct = pd.crosstab(d.Ttype, d[col], normalize="index")
    display((100 * ct).round(1))
    ct.plot.bar(stacked=True, figsize=(11, 4), colormap="tab10")
    plt.title(f"{col} class per T-type"); plt.legend(bbox_to_anchor=(1, 1)); plt.show()

# ---- Balmer decrements per T-type, with Case B values
fig, ax = plt.subplots(1, 4, figsize=(22, 4.5))
for a, col, caseB in zip(ax, ["Ha_Hb", "Hg_Hb", "Hd_Hb", "AV_gas"], [2.86, 0.468, 0.259, 0]):
    sns.boxplot(data=d, x="Ttype", y=col, palette=TT_COLORS, ax=a, showfliers=False)
    a.axhline(caseB, c="k", ls="--")
plt.tight_layout(); plt.show()

# ---- Age-sensitive plane: Dn4000 vs Ca II K and vs EW(Ha)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].scatter(d.Dn4000, d.CaII_K, c=d.Ttype, cmap="RdYlBu", s=6, alpha=0.6); ax[0].set_ylabel("Ca II K EW [Å]")
sc = ax[1].scatter(d.Dn4000, d.log_EW_Ha, c=d.Ttype, cmap="RdYlBu", s=6, alpha=0.6); ax[1].set_ylabel("log EW(Hα)")
for a in ax: a.set_xlabel("Dn4000")
plt.colorbar(sc, ax=ax, label="T-type"); plt.show()

# ---- Summary tables: median value per T-type + detection rate (S/N >= 3) of each emission line
display(d.groupby("Ttype")[ABS_FEATS + EMI_FEATS + RATIOS].median().round(2).T)
detect = pd.DataFrame({k: d.groupby("Ttype")[f"SN_{k}"].apply(lambda s: (s >= 3).mean())
                       for k in EMISSION}).T
plt.figure(figsize=(12, 6)); sns.heatmap(detect, annot=True, fmt=".2f", cmap="viridis")
plt.title("Fraction of galaxies with each emission line detected (S/N ≥ 3)"); plt.show()
feat.to_csv(f"{SAVE_DIR}/features_with_diagnostics.csv")