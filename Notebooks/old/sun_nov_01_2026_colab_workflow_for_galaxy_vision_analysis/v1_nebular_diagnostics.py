detected_flux = (
    lines.loc[lines["detected"].fillna(False)]
    .pivot(index="name", columns="line", values="flux")
    .reindex(features["name"])
)

def line_flux(name):
    if name in detected_flux:
        return detected_flux[name]
    return pd.Series(np.nan, index=detected_flux.index)

ha = line_flux("Halpha")
hb = line_flux("Hbeta")
o3 = line_flux("OIII5007")
n2 = line_flux("NII6583")
s2 = line_flux("SII6716") + line_flux("SII6731")
o1 = line_flux("OI6300")
o2 = line_flux("OII3726") + line_flux("OII3729")

diagnostics = pd.DataFrame(index=detected_flux.index)
diagnostics["bpt_y"] = np.log10(o3 / hb)
diagnostics["bpt_NII_x"] = np.log10(n2 / ha)
diagnostics["bpt_SII_x"] = np.log10(s2 / ha)
diagnostics["bpt_OI_x"] = np.log10(o1 / ha)
diagnostics["Halpha_Hbeta"] = ha / hb
diagnostics["Hgamma_Hbeta"] = line_flux("Hgamma") / hb
diagnostics["Hdelta_Hbeta"] = line_flux("Hdelta") / hb

# PyNeb Case B ratios under explicitly assumed temperature/density.
hydrogen = pn.RecAtom("H", 1)
intrinsic_ha_hb = (
    hydrogen.getEmissivity(GAS_TEMPERATURE, GAS_DENSITY, label="3_2")
    / hydrogen.getEmissivity(GAS_TEMPERATURE, GAS_DENSITY, label="4_2")
)

extinction_law = CCM89(Rv=RV)

def k_lambda(angstrom):
    # CCM89 returns A(lambda)/A(V); multiply by Rv for A(lambda)/E(B-V).
    return float(extinction_law(angstrom * u.AA) * RV)

diagnostics["gas_ebv_signed"] = (
    2.5 / (k_lambda(4861.33) - k_lambda(6562.80))
    * np.log10(diagnostics["Halpha_Hbeta"] / intrinsic_ha_hb)
)

# Preserve negative values above as an uncertainty/model-mismatch diagnostic.
diagnostics["gas_ebv_nonnegative"] = diagnostics["gas_ebv_signed"].clip(lower=0)
diagnostics["gas_av_nonnegative"] = RV * diagnostics["gas_ebv_nonnegative"]

# Additional observed excitation ratios.
diagnostics["O32_observed"] = (
    line_flux("OIII4959") + o3
) / o2
diagnostics["R23_observed"] = (
    o2 + line_flux("OIII4959") + o3
) / hb
diagnostics["NeIII_OII_observed"] = line_flux("NeIII3869") / o2
diagnostics["SII_doublet_ratio"] = line_flux("SII6716") / line_flux("SII6731")

# NII BPT: respect the domains of the rational demarcation curves.
def classify_nii(x, y):
    if not np.isfinite(x) or not np.isfinite(y):
        return "unclassified"
    if x < 0.05 and y < 0.61 / (x - 0.05) + 1.3:
        return "star-forming"
    if x < 0.47 and y < 0.61 / (x - 0.47) + 1.19:
        return "composite"
    return "AGN-like"

def classify_sii(x, y):
    if not np.isfinite(x) or not np.isfinite(y):
        return "unclassified"
    if x < 0.32 and y < 0.72 / (x - 0.32) + 1.30:
        return "star-forming"
    return "Seyfert-like" if y >= 1.89 * x + 0.76 else "LINER-like"

def classify_oi(x, y):
    if not np.isfinite(x) or not np.isfinite(y):
        return "unclassified"
    if x < -0.59 and y < 0.73 / (x + 0.59) + 1.33:
        return "star-forming"
    return "Seyfert-like" if y >= 1.18 * x + 1.30 else "LINER-like"

for panel, classifier in [
    ("NII", classify_nii), ("SII", classify_sii), ("OI", classify_oi)
]:
    diagnostics[f"bpt_{panel}_class"] = [
        classifier(x, y)
        for x, y in zip(diagnostics[f"bpt_{panel}_x"], diagnostics["bpt_y"])
    ]

# Add line EWs as classification candidates, without overwriting line tables.
if "ew_emission" in lines:
    ew_table = (
        lines.loc[lines["detected"].fillna(False)]
        .pivot(index="name", columns="line", values="ew_emission")
        .add_prefix("emission_ew_")
    )
    diagnostics = diagnostics.join(ew_table)

features = features.merge(
    diagnostics.reset_index(), on="name", how="left", validate="one_to_one"
)
features.to_csv(OUTPUT_DIR / "galaxy_features_with_diagnostics.csv", index=False)

for panel in ["NII", "SII", "OI"]:
    counts = pd.crosstab(features["ttype"], features[f"bpt_{panel}_class"])
    fractions = counts.div(counts.sum(axis=1), axis=0)
    counts.to_csv(OUTPUT_DIR / f"BPT_{panel}_counts.csv")
    fractions.to_csv(OUTPUT_DIR / f"BPT_{panel}_fractions_all_galaxies.csv")

display(features[["name", "ttype", "Dn4000", "Halpha_Hbeta", "bpt_NII_class"]].head())