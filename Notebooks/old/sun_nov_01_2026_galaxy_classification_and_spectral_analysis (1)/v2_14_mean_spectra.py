mean_spec = {t: np.nanmean(SPEC[tt_of == t], axis=0) for t in tts}

fig, ax = plt.subplots(2, 1, figsize=(17, 11), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
for t in tts:
    ax[0].plot(GRID, mean_spec[t], color=TT_COLORS[t], lw=0.9,
               label=f"T={t} {TT_NAMES.get(t,'')} (N={np.sum(tt_of == t)})")
ax[0].set_ylabel("Mean F / F(5500)"); ax[0].legend(ncol=2, fontsize=8)
mark_lines(ax[0], ax[0].get_ylim()[1])

# Ratio to the earliest T-type -> shows only the differences (bluer continuum, emission lines)
ref = tts[0]
for t in tts[1:]:
    ax[1].plot(GRID, mean_spec[t] / mean_spec[ref], color=TT_COLORS[t], lw=0.7)
ax[1].axhline(1, c="k", lw=0.5); ax[1].set_ylabel(f"Ratio to T={ref}")
ax[1].set_xlabel("Rest-frame wavelength [Å]")
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/mean_spectra_by_ttype.png", dpi=130); plt.show()

pd.DataFrame(mean_spec, index=pd.Index(GRID, name="wave_rest")).to_csv(f"{SAVE_DIR}/mean_spectra_by_ttype.csv")