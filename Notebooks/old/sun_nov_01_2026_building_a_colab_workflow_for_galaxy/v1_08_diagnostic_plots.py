classes = sorted(df["ttype"].dropna().unique())

fig, ax = plt.subplots(figsize=(9, 7))
for morphology in classes:
    group = df[(df["ttype"] == morphology) & bpt_valid]
    ax.scatter(
        group["bpt_x"], group["bpt_y"],
        s=20, alpha=0.6, label=f"T={morphology:g} (n={len(group)})",
    )

xx = np.linspace(-2, 0.0, 400)
ax.plot(xx, 0.61 / (xx - 0.05) + 1.3, "--", color="black", label="Kauffmann 2003")
xx = np.linspace(-2, 0.4, 400)
ax.plot(xx, 0.61 / (xx - 0.47) + 1.19, "-", color="black", label="Kewley 2001")

ax.set(
    xlabel="log10([N II] / Hα)",
    ylabel="log10([O III] / Hβ)",
    xlim=(-2, 0.7), ylim=(-1.5, 1.5),
    title="Integrated [N II] BPT by catalog T-type",
)
ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1))
fig.savefig(OUTPUT_DIR / "bpt_all_ttypes.png", dpi=180, bbox_inches="tight")
plt.show()

# One diagnostic panel per exact morphology class.
for morphology in classes:
    group = df[df["ttype"] == morphology]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(group["bpt_x"], group["bpt_y"], s=20)
    xx = np.linspace(-2, 0, 300)
    axes[0].plot(xx, 0.61 / (xx - 0.05) + 1.3, "k--")
    xx = np.linspace(-2, 0.4, 300)
    axes[0].plot(xx, 0.61 / (xx - 0.47) + 1.19, "k-")
    axes[0].set(xlabel="log10([N II]/Hα)", ylabel="log10([O III]/Hβ)",
                xlim=(-2, 0.7), ylim=(-1.5, 1.5))

    decrement = group["ha_hb"].dropna()
    if len(decrement):
        axes[1].hist(decrement, bins=min(25, max(3, len(decrement))))
    axes[1].axvline(INTRINSIC_HA_HB, color="black", linestyle="--")
    axes[1].set(xlabel="Hα/Hβ", ylabel="Galaxy count")
    fig.suptitle(f"T={morphology:g}; all processed galaxies n={len(group)}")
    fig.savefig(OUTPUT_DIR / f"diagnostics_T{morphology:g}.png", dpi=160)
    plt.close(fig)