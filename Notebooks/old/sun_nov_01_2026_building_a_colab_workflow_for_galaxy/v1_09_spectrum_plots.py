fig_mean, ax_mean = plt.subplots(figsize=(15, 6))
stack_tables = []

for morphology in classes:
    group = df[
        (df["ttype"] == morphology) & df["normalization_valid"]
    ]
    ids = group["galaxy_id"].tolist()
    if not ids:
        continue

    # Work with one morphology class at a time to limit memory use.
    spectra = np.stack([
        np.load(cache_dir / f"{identity}.npz")["flux"] for identity in ids
    ])

    contributors = np.isfinite(spectra).sum(axis=0)
    sums = np.nansum(spectra, axis=0)
    mean = np.divide(
        sums, contributors,
        out=np.full(len(REST_GRID), np.nan),
        where=contributors > 0,
    )

    ax_mean.plot(
        REST_GRID, mean,
        label=f"T={morphology:g}, n={len(ids)}",
    )
    stack_tables.append(pd.DataFrame({
        "ttype": morphology,
        "rest_wave_angstrom": REST_GRID,
        "mean_normalized_flux": mean,
        "contributors": contributors,
    }))

    # Overview includes ALL normalized spectra in this morphology class.
    fig, ax = plt.subplots(figsize=(14, 5))
    for spectrum in spectra:
        ax.plot(REST_GRID, spectrum, color="steelblue", alpha=0.08, linewidth=0.5)
    ax.plot(REST_GRID, mean, color="black", linewidth=1.5, label="Equal-galaxy mean")
    ax.set(
        xlabel="Rest-frame wavelength [Å]", ylabel="Normalized fλ",
        title=f"All normalized spectra: T={morphology:g}, n={len(ids)}",
    )
    ax.legend()
    fig.savefig(OUTPUT_DIR / f"all_spectra_T{morphology:g}.png", dpi=180)
    plt.close(fig)

    # Paginated views make every galaxy individually inspectable.
    for start in range(0, len(ids), 12):
        fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
        for local_index, ax in enumerate(axes.flat):
            index = start + local_index
            if index >= len(ids):
                ax.axis("off")
                continue
            ax.plot(REST_GRID, spectra[index], linewidth=0.6)
            ax.set_title(ids[index], fontsize=9)
        fig.suptitle(f"T={morphology:g}: galaxies {start + 1}–{min(start + 12, len(ids))}")
        fig.supxlabel("Rest-frame wavelength [Å]")
        fig.supylabel("Normalized fλ")
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"spectra_T{morphology:g}_page{start // 12 + 1}.png",
            dpi=140,
        )
        plt.close(fig)

ax_mean.set(
    xlabel="Rest-frame wavelength [Å]",
    ylabel="Mean normalized fλ",
    title="Mean spectra of all observed catalog T-types",
)
ax_mean.legend(fontsize=9, bbox_to_anchor=(1.02, 1))
fig_mean.tight_layout()
fig_mean.savefig(OUTPUT_DIR / "mean_spectra_all_ttypes.png", dpi=180, bbox_inches="tight")
plt.show()

if stack_tables:
    pd.concat(stack_tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "class_mean_spectra.csv", index=False
    )

display(df.groupby("ttype")["normalization_valid"].agg(["count", "sum"]))