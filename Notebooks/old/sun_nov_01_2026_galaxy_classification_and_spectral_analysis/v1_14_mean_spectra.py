fig, ax = plt.subplots(2, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
mean_spec = {}
for c in CLASS_ORDER:
    S_c = SPEC[cls_of == c]
    mean_spec[c] = np.nanmean(S_c, axis=0)
    ax[0].plot(GRID, mean_spec[c], color=CLASS_COLORS[c], lw=1, label=f"{c} (N={len(S_c)})")
ax[0].set_ylabel("Mean F / F(5500)"); ax[0].legend(); mark_lines(ax[0], ax[0].get_ylim()[1])

# Bottom panel: each class divided by the elliptical mean -> shows only the differences
for c in CLASS_ORDER[1:]:
    ax[1].plot(GRID, mean_spec[c] / mean_spec["E"], color=CLASS_COLORS[c], lw=0.8, label=f"{c} / E")
ax[1].axhline(1, c="k", lw=0.5); ax[1].set_ylabel("Ratio to E"); ax[1].legend(ncol=5)
ax[1].set_xlabel("Rest-frame wavelength [Å]")
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/mean_spectra.png", dpi=130); plt.show()

pd.DataFrame(mean_spec, index=pd.Index(GRID, name="wave_rest")).to_csv(f"{SAVE_DIR}/mean_spectra.csv")