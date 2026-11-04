EM_MARK  = {"[OII]":3727, "[NeIII]":3869, "Hδ":4102, "Hγ":4340, "[OIII]4363":4363, "Hβ":4861,
            "[OIII]5007":5007, "HeI":5876, "[OI]":6300, "[NII]":6548, "Hα":6563, "[NII] ":6584,
            "[SII]":6724, "[ArIII]":7136}
ABS_MARK = {"K":3934, "H":3969, "G":4304, "Mg":5175, "NaD":5893, "CaII":8498, "CaII ":8542, "CaII  ":8662}

def mark_lines(ax, ytop):
    for lab, w in EM_MARK.items():
        ax.axvline(w, c="tab:blue", lw=0.4, ls=":")
        ax.text(w, ytop, lab.strip(), fontsize=6, rotation=90, va="top", color="tab:blue")
    for lab, w in ABS_MARK.items():
        ax.axvline(w, c="tab:red", lw=0.4, ls=":")
        ax.text(w, ytop, lab.strip(), fontsize=6, rotation=90, va="top", color="tab:red")

tt_of = feat["Ttype"].reindex(spec_index).values
tts = sorted(pd.unique(tt_of[~pd.isna(tt_of)]))
fig, axes = plt.subplots(len(tts), 1, figsize=(16, 2.8 * len(tts)), sharex=True)
for ax, t in zip(np.atleast_1d(axes), tts):
    S_t = SPEC[tt_of == t]
    for s in S_t:                                    # every galaxy as a faint line
        ax.plot(GRID, s, color=TT_COLORS[t], lw=0.3, alpha=0.05, rasterized=True)
    ax.plot(GRID, np.nanmedian(S_t, axis=0), "k", lw=0.9)   # median on top
    ax.set_ylim(0, 3); ax.set_ylabel("F/F(5500)")
    ax.set_title(f"T-type {t} ({TT_NAMES.get(t,'')}), N = {len(S_t)}", loc="left", fontsize=10)
    mark_lines(ax, 2.95)
axes[-1].set_xlabel("Rest-frame wavelength [Å]")
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/all_spectra_by_ttype.png", dpi=100); plt.show()