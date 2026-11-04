def detected(frame, name):
    column = f"{name}_detected"
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)


def line_flux(frame, name):
    return pd.to_numeric(
        frame.get(f"{name}_flux", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )


bpt_valid = (
    df["continuum_trusted"].fillna(False)
    & detected(df, "ha") & detected(df, "hb")
    & detected(df, "nii6583") & detected(df, "oiii5007")
)

df["bpt_x"] = np.nan
df["bpt_y"] = np.nan
df.loc[bpt_valid, "bpt_x"] = np.log10(
    line_flux(df, "nii6583")[bpt_valid] / line_flux(df, "ha")[bpt_valid]
)
df.loc[bpt_valid, "bpt_y"] = np.log10(
    line_flux(df, "oiii5007")[bpt_valid] / line_flux(df, "hb")[bpt_valid]
)

df["bpt_class"] = "unclassified"
x, y = df["bpt_x"], df["bpt_y"]

# Domain guards avoid evaluating the demarcations across their poles.
sf = bpt_valid & (x < 0.05) & (y < 0.61 / (x - 0.05) + 1.3)
agn = bpt_valid & (
    (x >= 0.47) | ((x < 0.47) & (y >= 0.61 / (x - 0.47) + 1.19))
)

df.loc[bpt_valid, "bpt_class"] = "composite"
df.loc[sf, "bpt_class"] = "star_forming"
df.loc[agn, "bpt_class"] = "AGN_like"

# [N II] BPT alone does not robustly distinguish Seyfert from LINER-like gas.
# WHAN and [S II]/[O I] diagnostics are complementary extensions.

balmer_valid = (
    df["continuum_trusted"].fillna(False)
    & detected(df, "ha") & detected(df, "hb")
)
df["ha_hb"] = np.nan
df.loc[balmer_valid, "ha_hb"] = (
    line_flux(df, "ha")[balmer_valid] / line_flux(df, "hb")[balmer_valid]
)

# Case-B Halpha/Hbeta ~2.86 assumes ~10^4 K and low density.
# A different intrinsic ratio can be appropriate for AGN-like excitation.
INTRINSIC_HA_HB = 2.86
RV = 3.1
law = CCM89(Rv=RV)
k_ha = float(law(6564.61 * u.AA)) * RV
k_hb = float(law(4862.68 * u.AA)) * RV

df["nebular_ebv_approx"] = (
    2.5 / (k_hb - k_ha) * np.log10(df["ha_hb"] / INTRINSIC_HA_HB)
)
# Keep negative values visible: they can expose noise or modeling problems.
df["balmer_below_case_b"] = df["ha_hb"] < INTRINSIC_HA_HB

df.to_csv(OUTPUT_DIR / "galaxy_features.csv", index=False)

summary = pd.crosstab(df["ttype"], df["bpt_class"])
summary.to_csv(OUTPUT_DIR / "bpt_counts_by_ttype.csv")
display(summary)