packages = [
    "astropy", "dust_extinction", "numpy", "pandas",
    "scipy", "matplotlib", "scikit-learn",
]
provenance = {
    "versions": {package: metadata.version(package) for package in packages},
    "random_seed": RANDOM_SEED,
    "normalization_band_angstrom": NORMALIZATION_BAND,
    "wavelength_convention": WAVELENGTH_CONVENTION,
    "starlight_frame": STARLIGHT_FRAME,
    "starlight_input_has_flux_jacobian": STARLIGHT_INPUT_HAS_FLUX_JACOBIAN,
    "ignored_error_column": "ERROR_FoV",
    "line_uncertainties": "Approximate local-residual conditional covariance",
    "processed_galaxies": len(df),
    "max_galaxies_configuration": MAX_GALAXIES,
}
(OUTPUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))