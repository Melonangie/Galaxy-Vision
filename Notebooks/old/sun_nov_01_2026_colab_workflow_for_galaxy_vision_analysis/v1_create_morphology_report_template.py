report_template = """# Galaxy Morphology and Spectral Feature Interpretation

> Status: TEMPLATE — no scientific findings are implied.
> Replace bracketed prompts with measured results.
> Distinguish exploratory associations from validated predictive performance.

## 1. Executive summary

**Research question:** Which integrated spectral and STARLIGHT features
best separate the catalog morphological Ttypes in this sample?

- Galaxies discovered: [N]
- Galaxies with usable rest-frame spectra: [N]
- Galaxies with validated STARLIGHT continua: [N]
- Morphological stages represented: [list catalog Ttypes and Type strings]
- Most informative features: [results, not prior expectations]
- Best-separated type pairs: [pairs, effect sizes, uncertainty]
- Most overlapping type pairs: [pairs and confusion evidence]
- Cross-validated balanced accuracy: [mean and fold variation]
- Cross-validated macro F1: [mean and fold variation]
- Main limitations: [coverage, labels, dust, aperture, fitting, selection]

**One-sentence conclusion:**
[State what the data support about morphology separation, without equating
spectral excitation classes with morphological types.]

## 2. Dataset, provenance and measurement conventions

### 2.1 Inputs and reproducibility

- Run date / analysis identifier: [...]
- Notebook version / commit: [...]
- Spectrum directory and file count: [...]
- STARLIGHT version, configuration and SSP base: [...]
- Supporting catalog version and source: [...]
- Configuration snapshot and dependency versions: [...]
- Output directory: [...]

The supporting catalog supplies labels, redshifts and contextual controls.
It is not treated as a separate standalone EDA dataset.

### 2.2 Spectral conventions

- Spectrum wavelength unit: [...]
- Spectrum wavelength medium: [air/vacuum; evidence]
- Original spectrum frame: [observed/rest; evidence]
- STARLIGHT wavelength medium/frame: [...]
- Flux-density unit and calibration evidence: [...]
- Rest-frame wavelength conversion: lambda_rest = lambda_obs / (1 + z)
- Flux-density coordinate convention: [...]
- Galactic foreground extinction treatment: [...]
- Internal gas attenuation treatment: [...]
- Shape-normalization band: [...]
- Emission EW convention: positive emission, rest-frame Angstrom
- Absorption EW convention: positive absorption, rest-frame Angstrom

ERROR_FoV was excluded from all measurements, weights and uncertainties.
Empirical fit errors are provisional and do not include all correlated-pixel
or stellar-continuum uncertainties.

## 3. Morphology labels and sample selection

### 3.1 Catalog conventions

Explain what each observed Ttype means in THIS catalog.
Preserve ambiguous codes and unusual Type strings.

| Ttype | Associated catalog Type strings | N discovered | N usable | Label caveat |
|---|---|---:|---:|---|
| [...] | [...] | [...] | [...] | [...] |

Do not automatically interpret Ttype=11 as irregular: the supplied sample
includes Type='S' at that code.

### 3.2 Sample flow and exclusions

| Stage / exclusion | Count | Reason | Dependence on Ttype |
|---|---:|---|---|
| Spectrum inventory | [...] | [...] | [...] |
| Duplicate identifiers | [...] | [...] | [...] |
| Missing catalog matches | [...] | [...] | [...] |
| Invalid redshifts | [...] | [...] | [...] |
| Unusable spectral coverage | [...] | [...] | [...] |
| Missing/invalid STARLIGHT fit | [...] | [...] | [...] |
| Failed normalization | [...] | [...] | [...] |
| Usable BPT inputs | [...] | [...] | [...] |

Describe whether exclusions disproportionately remove particular stages.

## 4. Spectral and continuum quality assessment

- STARLIGHT alignment checks and continuum mismatch: [...]
- Fit inspection sample per Ttype: [...]
- Sky/telluric contamination handling: [...]
- Gaussian boundary fits / unusual profiles: [...]
- Empirical uncertainty estimation and validation: [...]
- Original CSV gaps versus actual FITS coverage: [...]
- Instrumental resolution and FoV velocity mixing: [...]

Report any diagnostics that remain unvalidated.
Do not treat a completed fit as proof of a reliable measurement.

## 5. Spectra grouped by morphology

Insert:
1. All normalized spectra for each Ttype.
2. A single graph containing one mean normalized spectrum per Ttype.
3. Wavelength-dependent contributor counts.
4. Selected line/absorption-region zooms.

### Interpretation

- Continuum-shape differences: [...]
- Dn4000 differences: [...]
- Balmer emission/absorption differences: [...]
- Metal/molecular absorption differences: [...]
- Within-type diversity and outliers: [...]
- Regions where changing contributor counts affect comparisons: [...]

Means use equal galaxy weights after continuum normalization.
Mean spectra are not substitutes for individual-galaxy diagnostics.

## 6. Stellar populations and absorption by Ttype

| Feature | Ttype comparison | N per type | Median / IQR | Effect size | Interpretation |
|---|---|---|---|---|---|
| Dn4000 | [...] | [...] | [...] | [...] | [...] |
| Light-weighted log age | [...] | [...] | [...] | [...] | [...] |
| Mass-weighted log age | [...] | [...] | [...] | [...] | [...] |
| Young light fraction | [...] | [...] | [...] | [...] | [...] |
| HdeltaA observed | [...] | [...] | [...] | [...] | [...] |
| HdeltaA model | [...] | [...] | [...] | [...] | [...] |
| Mg b / G band / Ca K | [...] | [...] | [...] | [...] | [...] |
| Na D | [...] | [...] | [...] | [...] | [...] |
| Ca II triplet | [...] | [...] | [...] | [...] | [...] |

Discuss:
- Age–metallicity–dust degeneracy.
- Emission infill in observed Balmer absorption.
- Model predictions versus direct measurements.
- Non-independence of population fractions.
- Na D stellar versus interstellar contributions.
- Exploratory versus published absorption bandpasses.
- Resolution matching and lack of Lick calibration.

## 7. Emission-line and excitation diagnostics

### 7.1 Line availability

| Line / blend | Ttype | N covered | N measured | N detected | Detection fraction denominator |
|---|---|---:|---:|---:|---|
| [...] | [...] | [...] | [...] | [...] | [...] |

Separate coverage failures, continuum failures and nondetections.
An unavailable measurement is not zero line emission.

### 7.2 BPT results

Insert NII, SII and OI BPT diagrams for each Ttype and all Ttypes combined.

| Ttype | N total | N with valid BPT inputs | SF | Composite | AGN-like / Seyfert-like | LINER-like | Unclassified |
|---|---:|---:|---:|---:|---:|---:|---:|
| [...] | [...] | [...] | [...] | [...] | [...] | [...] | [...] |

State whether fractions use ALL galaxies or only classifiable galaxies.
Do not combine these denominators silently.

Interpretation:
[Describe excitation distributions and their overlap across morphology.]

Caution:
BPT classes describe excitation, not appearance. AGN-like/LINER-like
ratios can involve diffuse gas, evolved stars, shocks and FoV mixing.
They do not establish nuclear activity by themselves.

### 7.3 Additional gas diagnostics

- O32 and NeIII/OII excitation trends: [...]
- R23 behavior and calibration degeneracy: [...]
- SII doublet resolution/reliability: [...]
- Density assumptions or reasons not to estimate density: [...]
- Weak HeI, NeIII, OI and ArIII detection limitations: [...]

## 8. Balmer decrement and attenuation

- Assumed Case-B temperature and density: [...]
- Intrinsic Halpha/Hbeta used: [...]
- Extinction law and Rv: [...]
- Stellar absorption correction validation: [...]

| Ttype | N valid decrements | Halpha/Hbeta median / IQR | Signed gas E(B-V) | Negative-value fraction |
|---|---:|---|---|---|
| [...] | [...] | [...] | [...] | [...] |

Discuss Hgamma/Hbeta and Hdelta/Hbeta as consistency checks.
Distinguish gas attenuation from STARLIGHT stellar AV.
Explain signed versus nonnegative-clipped reddening results.

Do not assume that higher attenuation proves later morphology:
inclination, gas geometry and selection can contribute.

## 9. Which features separate morphological types?

### 9.1 Overall associations

| Feature | Usable N | Types tested | Epsilon squared | FDR q | Coverage caveat |
|---|---:|---:|---:|---:|---|
| [...] | [...] | [...] | [...] | [...] | [...] |

Large samples can produce small p-values for weak separation.
Prioritize effect size, distribution overlap and stability.

### 9.2 Pairwise separation

| Type pair | Feature | N_a / N_b | Signed rank-biserial effect | Absolute effect | FDR q | Bootstrap interval, if calculated |
|---|---|---|---|---|---|---|
| [...] | [...] | [...] | [...] | [...] | [...] | [...] |

Effect sign depends on the ordering of the pair.
Describe overlapping neighboring stages, not just extreme early/late pairs.
Leave confidence intervals unavailable unless they were actually calculated.

### 9.3 Predictive validation

- Predictors included: [...]
- Labels, identifiers and label-derived variables excluded: [...]
- Galaxy-level duplicate/group handling: [...]
- Sparse classes excluded and reasons: [...]
- Cross-validation design: [...]
- Fold-local imputation/preprocessing: [...]
- Held-out permutation-importance method: [...]

| Feature set | Included types | N | Balanced accuracy | Macro F1 | Principal confusions |
|---|---|---:|---|---|---|
| Continuum + absorption | [...] | [...] | [...] | [...] | [...] |
| Emission | [...] | [...] | [...] | [...] | [...] |
| STARLIGHT | [...] | [...] | [...] | [...] | [...] |
| Combined spectroscopy | [...] | [...] | [...] | [...] | [...] |
| Optional catalog-augmented | [...] | [...] | [...] | [...] | [...] |

Insert the out-of-fold confusion matrix and permutation-importance ranking.

Distinguish:
- Statistical association.
- Out-of-sample predictive contribution.
- Physical interpretation.
- Causal claims, which this EDA does not establish.

Correlated predictors can share or dilute permutation importance.
Exploratory feature choices require independent or nested validation
before final performance claims.

## 10. Confounders and sensitivity analyses

| Potential confounder | Check performed | Result | Effect on conclusions |
|---|---|---|---|
| Stellar mass | [...] | [...] | [...] |
| Inclination / dust | [...] | [...] | [...] |
| Redshift / coverage | [...] | [...] | [...] |
| FoV / aperture | [...] | [...] | [...] |
| STARLIGHT fit quality | [...] | [...] | [...] |
| Emission detection selection | [...] | [...] | [...] |
| Continuum-normalization choice | [...] | [...] | [...] |
| Empirical-noise assumptions | [...] | [...] | [...] |
| Label uncertainty | [...] | [...] | [...] |

Catalog angular size alone does not establish the fraction of a galaxy
covered by the spectrum. Additional aperture information may be needed.

Report checks not performed as limitations, not as reassuring null results.

## 11. Morphology-focused synthesis

### What appears well supported?
[Measured differences, effect sizes and predictive evidence.]

### What remains ambiguous?
[E versus S0 overlap, neighboring spiral stages, bars, population degeneracy,
outliers or uncertain labels.]

### What can integrated spectra not establish?
[Direct spiral-arm structure, bulge/disk geometry, bars or interactions
without imaging or additional spatial information.]

### Recommended feature set
[Small, interpretable set with adequate coverage and validation.]

### Recommended next steps
[Imaging comparison, improved uncertainty model, continuum validation,
resolution matching, independent test sample or spatially resolved analysis.]

## 12. Limitations

Explicitly address:
- Incorrect supplied ERROR_FoV.
- Provisional STARLIGHT-derived quantities.
- Empirical errors and approximate upper limits.
- Nonnegative weak-line fitting bias.
- Unresolved doublets and blend-detection limitations.
- Missing features and morphology-dependent selection.
- Flux calibration, foreground extinction and wavelength conventions.
- Instrumental resolution and integrated-aperture mixing.
- Sparse/imbalanced classes and catalog-label uncertainty.
- Lack of independent final validation, if applicable.

## 13. References

- Buta, Galaxy Morphology:
  https://arxiv.org/abs/1102.0550
- RC3 classification documentation:
  https://ned.ipac.caltech.edu/level5/Sept02/RC3/RC3_contents.html
- Balogh et al. 1999, Dn4000:
  https://ui.adsabs.harvard.edu/abs/1999ApJ...527...54B/abstract
- Worthey & Ottaviani 1997, Balmer absorption indices:
  https://ui.adsabs.harvard.edu/abs/1997ApJS..111..377W/abstract
- Bruzual & Charlot 2003, stellar-population synthesis:
  https://arxiv.org/abs/astro-ph/0309134
- Cid Fernandes et al. 2005, STARLIGHT:
  https://arxiv.org/abs/astro-ph/0412481
- Baldwin, Phillips & Terlevich 1981:
  https://ui.adsabs.harvard.edu/abs/1981PASP...93....5B/abstract
- Kewley et al. 2001:
  https://arxiv.org/abs/astro-ph/0106324
- Kauffmann et al. 2003:
  https://arxiv.org/abs/astro-ph/0304239
- Kewley et al. 2006:
  https://arxiv.org/abs/astro-ph/0605681
- Cid Fernandes et al. 2011, WHAN:
  https://arxiv.org/abs/1012.4426
- Cardelli, Clayton & Mathis 1989:
  https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract
- Luridiana et al. 2015, PyNeb:
  https://arxiv.org/abs/1410.6662

## Appendix: Output index

- feature_data_dictionary.csv
- galaxy_features_with_diagnostics.csv
- emission_lines.csv
- quality_control.csv
- feature_summary_by_ttype.csv
- feature_available_fraction_by_ttype.csv
- overall_feature_separation.csv
- pairwise_feature_separation.csv
- classification_cv_scores.csv [if calculated]
- classification_feature_ranking.csv [if calculated]
- mean_spectra_by_ttype.csv
- figures/
"""

report_path = OUTPUT_DIR / "morphology_interpretation_report_template.md"

# Avoid overwriting a report that you may already have edited.
if report_path.exists():
    print("Template already exists; not overwritten:", report_path)
else:
    report_path.write_text(report_template, encoding="utf-8")
    print("Saved:", report_path)