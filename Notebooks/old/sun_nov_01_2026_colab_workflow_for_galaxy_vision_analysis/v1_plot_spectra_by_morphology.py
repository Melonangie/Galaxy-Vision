means = {}
contributor_counts = {}

fig_mean, ax_mean = plt.subplots(figsize=(16, 6))

for ttype, group in features.dropna(subset=["ttype"]).groupby("ttype"):
    fig, ax = plt.subplots(figsize=(16, 5))
    total = np.zeros(len(GRID), dtype=float)
    count = np.zeros(len(GRID), dtype=int)
    included = 0

    for name in group["name"]:
        cache = OUTPUT_DIR / "spectra_cache" / f"{name}.npz"
        if not cache.exists():
            continue

        with np.load(cache) as saved:
            spectrum = saved["normalized_flux"]

        valid = np.isfinite(spectrum)
        total[valid] += spectrum[valid]
        count[valid] += 1
        included += 1

        # Rasterize the many curves so saved plots remain practical.
        ax.plot(GRID, spectrum, color="steelblue", alpha=0.08,
                linewidth=0.4, rasterized=True)

    if included == 0:
        plt.close(fig)
        continue

    mean = np.divide(total, count, out=np.full(len(GRID), np.nan), where=count > 0)
    means[ttype] = mean
    contributor_counts[ttype] = count

    ax.plot(GRID, mean, color="black", linewidth=1.2, label="Mean")
    ax.set(
        title=f"Ttype {ttype}: all {included} normalized spectra",
        xlabel="Rest-frame air wavelength [Å]",
        ylabel="Flux / normalization-band mean",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "figures" / f"all_spectra_Ttype_{ttype}.png", dpi=180
    )
    plt.show()
    plt.close(fig)

    ax_mean.plot(GRID, mean, linewidth=1,
                 label=f"T={ttype}, N={included}")

ax_mean.set(
    title="Mean normalized spectrum of every Ttype",
    xlabel="Rest-frame air wavelength [Å]",
    ylabel="Mean normalized flux",
)
ax_mean.legend(ncol=3)
fig_mean.tight_layout()
fig_mean.savefig(
    OUTPUT_DIR / "figures" / "mean_spectra_all_ttypes.png", dpi=180
)
plt.show()
plt.close(fig_mean)

# Contributor counts are essential: edge regions can contain fewer galaxies.
fig, ax = plt.subplots(figsize=(16, 4))
for ttype, count in contributor_counts.items():
    ax.plot(GRID, count, label=f"T={ttype}")
ax.set(
    title="Number of galaxies contributing at each wavelength",
    xlabel="Rest-frame air wavelength [Å]",
    ylabel="Contributing galaxies",
)
ax.legend(ncol=3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "figures" / "stack_contributor_counts.png", dpi=160)
plt.show()
plt.close(fig)

stack_table = pd.DataFrame({"wave_air_rest_A": GRID})
for ttype in means:
    stack_table[f"mean_T{ttype}"] = means[ttype]
    stack_table[f"count_T{ttype}"] = contributor_counts[ttype]
stack_table.to_csv(OUTPUT_DIR / "mean_spectra_by_ttype.csv", index=False)