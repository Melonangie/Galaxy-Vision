# Main lines to mark on the plots (emission = blue, absorption = red)
MARK_EM  = {"[OII]": 3727, "Hβ": 4861, "[OIII]": 5007, "[OI]": 6300, "Hα+[NII]": 6563, "[SII]": 6724}
MARK_ABS = {"CaII K": 3934, "CaII H": 3969, "G band": 4304, "Mgb": 5175, "NaD": 5893}

def mark_lines(ax, ymax):
    for lab, w in MARK_EM.items():
        ax.axvline(w, c="tab:blue", lw=0.5, ls=":"); ax.text(w, ymax, lab, fontsize=7, rotation=90, va="top", color="tab:blue")
    for lab, w in MARK_ABS.items():
        ax.axvline(w, c="tab:red", lw=0.5, ls=":"); ax.text(w, ymax, lab, fontsize=7, rotation=90, va="top", color="tab:red")

cls_of = feat["morph_class"].reindex(spec_index).values
fig, axes = plt.subplots(len(CLASS_ORDER), 1, figsize=(16, 3.2 * len(CLASS_ORDER)), sharex=True)
for ax, c in zip(axes, CLASS_ORDER):
    S_c = SPEC[cls_of == c]
    for s in S_c:                                    # every galaxy: thin, transparent line
        ax.plot(GRID, s, color=CLASS_COLORS[c], lw=0.3, alpha=0.05, rasterized=True)
    ax.plot(GRID, np.nanmedian(S_c, axis=0), "k", lw=1, label="median")
    ax.set_ylim(0, 3); ax.set_ylabel("F / F(5500)")
    ax.set_title(f"{c}  (N = {len(S_c)})", loc="left"); mark_lines(ax, 2.95)
axes[-1].set_xlabel("Rest-frame wavelength [Å]")
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/all_spectra_by_class.png", dpi=110); plt.show()