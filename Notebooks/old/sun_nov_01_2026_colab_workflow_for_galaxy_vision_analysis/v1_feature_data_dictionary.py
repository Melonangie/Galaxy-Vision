# Run after the previous notebook cells.
# Requires OUTPUT_DIR and the earlier feature/line definitions.

dictionary_rows = []

def add_field(dataset, field, unit, origin, definition, role, caveat=""):
    """One readable dictionary entry per output column."""
    dictionary_rows.append({
        "dataset": dataset,
        "field": field,
        "unit": unit,
        "origin": origin,
        "definition": definition,
        "modeling_role": role,
        "caveat": caveat,
    })


# ------------------------------------------------------------
# A. Identifiers, morphology labels and supporting information
# ------------------------------------------------------------
metadata_fields = [
    ("name", "none", "Canonical manga-<plate>-<ifu> galaxy identifier.",
     "identifier; exclude from predictors", "Must be unique per galaxy."),
    ("ttype", "catalog code", "Morphological stage supplied by the catalog.",
     "target; exclude from predictors",
     "Preserve catalog conventions; do not assume Ttype=11 means irregular."),
    ("type", "catalog label", "Catalog morphology string, such as E, S0 or SBab.",
     "label; exclude from predictors",
     "Contains target information, including structural family labels."),
    ("nsa_z", "dimensionless", "Catalog redshift used for rest-frame conversion.",
     "context/control; not a primary morphology feature",
     "Missing/sentinel redshifts cannot be imputed for spectral registration."),
    ("logmass", "catalog-defined logarithmic mass",
     "Supporting catalog LogMass value.",
     "optional augmented predictor/control",
     "Verify mass units, logarithm base, IMF and cosmological convention."),
    ("incl", "catalog-defined; verify degrees",
     "Supporting catalog inclination estimate.",
     "context/control",
     "Inclination affects dust, projected structure and FoV velocity mixing."),
    ("petro_th90", "catalog-defined angular size",
     "Supporting catalog PETRO_TH90 value.",
     "context/control",
     "Verify band and units; this alone does not measure FoV coverage fraction."),
]

for field, unit, definition, role, caveat in metadata_fields:
    add_field("galaxy_features", field, unit, "supporting catalog",
              definition, role, caveat)


# ------------------------------------------------------------
# B. Stellar populations and STARLIGHT fit diagnostics
# ------------------------------------------------------------
population_fields = [
    ("light_log_age", "log10(year)",
     "sum(normalized x_j * log10(age_j / year))",
     "Luminous young stars can dominate this mean."),
    ("mass_log_age", "log10(year)",
     "sum(normalized Mcor_j * log10(age_j / year))",
     "Uses current stellar mass fractions, not initial mass fractions."),
    ("light_Z", "SSP metallicity mass fraction",
     "sum(normalized x_j * Z_j)",
     "Linear metallicity mean; not log(Z/Z_sun) or gas metallicity."),
    ("mass_Z", "SSP metallicity mass fraction",
     "sum(normalized Mcor_j * Z_j)",
     "Linear metallicity mean; depends on the adopted SSP library."),
    ("young_light_fraction", "fraction",
     "sum(normalized x_j for age_j < 1e8 year)",
     "Young/intermediate/old boundaries are analysis choices."),
    ("intermediate_light_fraction", "fraction",
     "sum(normalized x_j for 1e8 <= age_j < 1e9 year)",
     "A coarse population summary, not a resolved star-formation history."),
    ("old_light_fraction", "fraction",
     "sum(normalized x_j for age_j >= 1e9 year)",
     "Correlated with the other fractions; fractions sum to one."),
    ("stellar_av", "mag",
     "STARLIGHT AV_min stellar attenuation parameter",
     "Not interchangeable with gas attenuation from the Balmer decrement."),
]

population_caveat = (
    "Model-dependent and provisional if upstream uncertainties were incorrect. "
    "Population weights are renormalized before averaging."
)

for field, unit, definition, caveat in population_fields:
    add_field("galaxy_features", field, unit, "STARLIGHT best-model output",
              definition, "physical feature candidate",
              caveat + " " + population_caveat)

for field, unit, definition, caveat in [
    ("starlight_adev", "percent", "STARLIGHT adev fit-deviation statistic.",
     "Fit quality, not a physical morphology measurement."),
    ("starlight_chi2", "dimensionless", "STARLIGHT chi2/Nl_eff statistic.",
     "Not a calibrated reduced chi-square if the input errors are wrong."),
    ("stellar_vdisp_fit", "km/s", "STARLIGHT vd_min fitted broadening.",
     "Requires instrumental/template-resolution validation before physical use."),
]:
    add_field("galaxy_features", field, unit, "STARLIGHT best-model output",
              definition,
              "QC" if field != "stellar_vdisp_fit"
              else "provisional physical candidate",
              caveat)


# ------------------------------------------------------------
# C. Continuum and absorption features
# ------------------------------------------------------------
add_field(
    "galaxy_features", "Dn4000", "dimensionless", "original spectrum",
    "Mean f_nu in 4000–4100 Å divided by mean f_nu in 3850–3950 Å; "
    "f_nu is proportional to f_lambda * wavelength^2.",
    "physical feature candidate",
    "Age–metallicity degeneracy. The current implementation averages over "
    "available endpoint samples within each fully covered band."
)

add_field(
    "galaxy_features", "continuum_red_blue", "dimensionless",
    "original spectrum",
    "Mean f_lambda(6000–6100 Å) / mean f_lambda(4500–4600 Å).",
    "physical feature candidate",
    "Affected by stellar populations, attenuation and flux calibration."
)

absorption_meanings = {
    "CaK": "Ca II K absorption near 3933.66 Å.",
    "CaH_blend": "Ca II H region near 3968.47 Å, blended with H epsilon.",
    "Gband": "Broad CH molecular G-band absorption near 4300 Å.",
    "Mgb": "Mg b triplet region near 5167, 5173 and 5184 Å.",
    "NaD": "Na D doublet region near 5890 and 5896 Å.",
    "CaII8498": "Near-infrared Ca II triplet component near 8498 Å.",
    "CaII8542": "Near-infrared Ca II triplet component near 8542 Å.",
    "CaII8662": "Near-infrared Ca II triplet component near 8662 Å.",
    "HdeltaA": "H delta A absorption index using published bandpasses.",
}

# Use the exact band definitions from the measurement cell.
for index, bands in ABSORPTION_BANDS.items():
    feature_band, blue_band, red_band = bands

    for origin in ["observed", "stellar_model"]:
        add_field(
            "galaxy_features", f"{index}_{origin}_ew", "rest-frame Å",
            "original spectrum" if origin == "observed"
            else "validated, rescaled STARLIGHT synthetic spectrum",
            f"{absorption_meanings[index]} "
            f"Integral of (1 - flux/pseudo_continuum) dλ. "
            f"Feature={feature_band}; blue={blue_band}; red={red_band} Å.",
            "physical feature candidate" if origin == "observed"
            else "model-derived feature candidate",
            "Positive means absorption. Observed indices may contain emission "
            "infill; model indices are predictions, not independent observations. "
            "Except HdeltaA, these are exploratory bandpasses; none is claimed "
            "to be resolution-matched or Lick-calibrated. Na D may include ISM "
            "absorption; near-IR Ca lines require sky/telluric checks."
        )

add_field(
    "galaxy_features", "CaII_triplet_observed_ew", "rest-frame Å",
    "original spectrum",
    "Sum of CaII8498, CaII8542 and CaII8662 observed absorption EWs, "
    "only when all three indices are available.",
    "physical feature candidate",
    "Missing components yield NaN, not a partial triplet sum."
)


# ------------------------------------------------------------
# D. Emission-line table and wide EW features
# ------------------------------------------------------------
line_definitions = [
    ("line", "none", "Unique component name, such as OIII5007.", "identifier"),
    ("group", "none", "Joint fitting group, such as Halpha_NII.", "identifier"),
    ("flux", "input flux-density unit × Å",
     "Integrated fitted Gaussian flux after stellar-continuum subtraction.",
     "measurement; prefer ratios/EWs for morphology comparisons"),
    ("empirical_error", "same as flux",
     "Conditional flux-error approximation from the fit Jacobian and "
     "empirical sideband noise.", "uncertainty/QC"),
    ("snr_empirical", "dimensionless",
     "Fitted flux / empirical_error.", "detection/QC"),
    ("ew_emission", "rest-frame Å",
     "Fitted line flux / underlying stellar continuum at the line center.",
     "physical feature candidate"),
    ("detected", "boolean",
     "Empirical S/N passes DETECTION_SNR and velocity/width fit is "
     "not flagged at a boundary.", "selection/QC"),
    ("upper_limit_approx", "same as flux",
     "max(fitted flux, 0) + DETECTION_SNR * empirical_error.",
     "approximate limit"),
    ("velocity_fit", "km/s",
     "Shared group velocity offset relative to adopted rest wavelengths.",
     "fit diagnostic"),
    ("width_fit", "km/s",
     "Shared Gaussian velocity sigma, not FWHM.", "fit diagnostic"),
    ("boundary_fit", "boolean",
     "Fitted velocity or width lies close to an imposed parameter bound.",
     "QC"),
    ("residual_rms_over_noise", "dimensionless",
     "Standard deviation of fit residuals / empirical sideband noise.",
     "QC"),
    ("status", "text",
     "'measured' or the reason a measurement was unavailable.", "QC"),
]

for field, unit, definition, role in line_definitions:
    add_field(
        "emission_lines", field, unit, "continuum-subtracted Gaussian fit",
        definition, role,
        "ERROR_FoV is never used. Errors/limits omit stellar-model uncertainty "
        "and pixel covariance; limits are not calibrated confidence bounds. "
        "Widths include instrumental broadening and aperture mixing."
    )

for field in ["name", "ttype"]:
    definition = next(row for row in dictionary_rows
                      if row["dataset"] == "galaxy_features"
                      and row["field"] == field)
    add_field("emission_lines", field, definition["unit"],
              definition["origin"], definition["definition"],
              definition["modeling_role"], definition["caveat"])

for components in EMISSION_GROUPS.values():
    for line, center in components:
        add_field(
            "galaxy_features", f"emission_ew_{line}", "rest-frame Å",
            "detected emission-line fit",
            f"Emission-positive EW of {line}; adopted air center {center} Å.",
            "physical feature candidate",
            "Stored only for detected lines. NaN does not mean zero emission."
        )


# ------------------------------------------------------------
# E. Nebular ratios and excitation classes
# ------------------------------------------------------------
ratio_definitions = {
    "bpt_y": "log10([OIII]5007 / Hbeta)",
    "bpt_NII_x": "log10([NII]6583 / Halpha)",
    "bpt_SII_x": "log10(([SII]6716 + [SII]6731) / Halpha)",
    "bpt_OI_x": "log10([OI]6300 / Halpha)",
    "Halpha_Hbeta": "Halpha / Hbeta",
    "Hgamma_Hbeta": "Hgamma / Hbeta",
    "Hdelta_Hbeta": "Hdelta / Hbeta",
    "O32_observed": "([OIII]4959 + [OIII]5007) / ([OII]3726 + [OII]3729)",
    "R23_observed": "([OII]3726 + [OII]3729 + [OIII]4959 + [OIII]5007) / Hbeta",
    "NeIII_OII_observed": "[NeIII]3869 / ([OII]3726 + [OII]3729)",
    "SII_doublet_ratio": "[SII]6716 / [SII]6731",
}

for field, definition in ratio_definitions.items():
    add_field(
        "galaxy_features", field, "dimensionless", "detected emission-line fluxes",
        definition, "physical feature candidate",
        "Observed, not internally dust-corrected. All required components must "
        "be detected in the current code. This conservatively excludes some "
        "detectable blends. BPT measures excitation, not morphology; R23 is "
        "not a unique metallicity estimate. SII density requires resolution "
        "and temperature assumptions."
    )

for panel in ["NII", "SII", "OI"]:
    add_field(
        "galaxy_features", f"bpt_{panel}_class", "category",
        "BPT ratios and literature demarcations",
        f"Excitation class from the {panel} BPT panel; unavailable inputs "
        "produce 'unclassified'.",
        "derived excitation descriptor; not a morphology target",
        "AGN-like/LINER-like does not prove nuclear activity. Avoid including "
        "both ratios and derived classes without checking redundancy."
    )

for field, definition in [
    ("gas_ebv_signed",
     "2.5 / (k_Hbeta - k_Halpha) * "
     "log10(observed Halpha/Hbeta / assumed PyNeb Case-B ratio)"),
    ("gas_ebv_nonnegative", "max(gas_ebv_signed, 0)"),
    ("gas_av_nonnegative", "RV * gas_ebv_nonnegative"),
]:
    add_field(
        "galaxy_features", field, "mag", "Balmer decrement + extinction law",
        definition, "model-derived physical feature candidate",
        "Assumes configured gas temperature/density and CCM89 Rv. Negative "
        "signed values are preserved as diagnostics; clipping creates a "
        "pile-up at zero. Gas attenuation is distinct from stellar AV."
    )


# ------------------------------------------------------------
# F. Quality-control output
# ------------------------------------------------------------
for field, definition in [
    ("name", "Canonical galaxy identifier."),
    ("path", "Input spectrum path."),
    ("status", "Overall processing outcome or failure reason."),
    ("starlight_status", "Missing, duplicate, aligned or model-validation failure."),
    ("continuum_mismatch",
     "Maximum absolute fractional deviation of continuum-band scale ratios "
     "from their median ratio."),
    ("stack_status", "Spectrum-cache outcome or normalization failure."),
]:
    add_field(
        "quality_control", field,
        "fraction" if field == "continuum_mismatch" else "text",
        "processing pipeline", definition, "QC; exclude from physical predictors"
    )


# ------------------------------------------------------------
# G. Include every actual output column; flag unreviewed additions
# ------------------------------------------------------------
known = {(row["dataset"], row["field"]) for row in dictionary_rows}

for dataset, variable in [
    ("galaxy_features", "features"),
    ("emission_lines", "lines"),
    ("quality_control", "qc"),
]:
    table = globals().get(variable)
    if isinstance(table, pd.DataFrame):
        for field in table.columns:
            if (dataset, field) not in known:
                add_field(
                    dataset, field, "REVIEW", "REVIEW",
                    "Column exists but has not yet been documented.",
                    "exclude until reviewed",
                    "Add an explicit definition before interpreting or modeling."
                )

dictionary = pd.DataFrame(dictionary_rows)
dictionary["missing_value_policy"] = (
    "Numeric NaN = unavailable/invalid/not detected, depending on field; "
    "consult line status and QC. Never automatically interpret NaN as zero."
)

dictionary.to_csv(OUTPUT_DIR / "feature_data_dictionary.csv", index=False)

display(dictionary[["dataset", "field", "unit", "modeling_role"]])
print("Saved:", OUTPUT_DIR / "feature_data_dictionary.csv")