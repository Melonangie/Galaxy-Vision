def draw_bpt_boundaries(ax, panel):
    if panel == "NII":
        x = np.linspace(-2, 0.04, 400)
        ax.plot(x, 0.61 / (x - 0.05) + 1.3, "--", color="black",
                label="Kauffmann 2003")
        x = np.linspace(-2, 0.46, 400)
        ax.plot(x, 0.61 / (x - 0.47) + 1.19, "-", color="black",
                label="Kewley 2001")
    elif panel == "SII":
        x = np.linspace(-2, 0.31, 400)
        ax.plot(x, 0.72 / (x - 0.32) + 1.30, color="black")
        x = np.linspace(-0.3, 0.5, 100)
        ax.plot(x, 1.89 * x + 0.76, "--", color="black")
    else:
        x = np.linspace(-2.5, -0.60, 400)
        ax.plot(x, 0.73 / (x + 0.59) + 1.33, color="black")
        x = np.linspace(-1.1, 0, 100)
        ax.plot(x, 1.18 * x + 1.30, "--", color="black")


def plot_bpt(data, title, filename, colored=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = {
        "NII": "[N II]6583 / Hα",
        "SII": "([S II]6716+6731) / Hα",
        "OI": "[O I]6300 / Hα",
    }

    for panel, ax in zip(["NII", "SII", "OI"], axes):
        valid = data.dropna(subset=[f"bpt_{panel}_x", "bpt_y"])
        sns.scatterplot(
            data=valid, x=f"bpt_{panel}_x", y="bpt_y",
            hue="ttype" if colored else None,
            palette="tab20" if colored else None,
            s=15, alpha=0.6, ax=ax,
        )
        draw_bpt_boundaries(ax, panel)
        ax.set(xlabel=f"log₁₀({labels[panel]})",
               ylabel="log₁₀([O III]5007 / Hβ)",
               ylim=(-1.5, 1.5), xlim=(-2.5, 0.6))
        ax.set_title(f"{panel}: {len(valid)} classified-input galaxies")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / filename, dpi=160)
    plt.show()
    plt.close(fig)


plot_bpt(features, "All morphological Ttypes", "BPT_all_ttypes.png", colored=True)

for ttype, group in features.dropna(subset=["ttype"]).groupby("ttype"):
    plot_bpt(group, f"Ttype {ttype}", f"BPT_Ttype_{ttype}.png")