feature_rows = []
line_rows = []
qc_rows = []

selected_paths = (
    spectrum_paths if MAX_GALAXIES is None
    else spectrum_paths[:MAX_GALAXIES]
)

for path in tqdm(selected_paths):
    name = galaxy_id(path)
    qc = {"name": name, "path": str(path)}

    try:
        if name not in catalog.index:
            raise ValueError("No supporting catalog match.")

        metadata = catalog.loc[name]
        z = metadata["nsa_z"]
        if not np.isfinite(z) or z < 0:
            raise ValueError("Missing/invalid redshift.")

        wavelength, flux = read_spectrum(path, z)
        row = {"name": name, "ttype": metadata["ttype"]}

        # Context only, not automatically physical morphology predictors.
        for column in ["type", "nsa_z", "logmass", "incl", "petro_th90"]:
            if column in metadata.index:
                row[column] = metadata[column]

        stellar_model = None
        matches = out_lookup.get(name, [])

        if len(matches) > 1:
            qc["starlight_status"] = "duplicate STARLIGHT files"
        elif len(matches) == 0:
            qc["starlight_status"] = "missing"
        else:
            try:
                model_wave, model_flux, stellar_features = read_starlight(matches[0], z)
                stellar_model, mismatch = align_stellar_model(
                    wavelength, flux, model_wave, model_flux
                )
                row.update(stellar_features)
                qc["starlight_status"] = "aligned"
                qc["continuum_mismatch"] = mismatch
            except Exception as error:
                qc["starlight_status"] = str(error)

        row.update(continuum_features(wavelength, flux, stellar_model))

        for group, components in EMISSION_GROUPS.items():
            try:
                if stellar_model is None:
                    raise ValueError("No validated stellar continuum.")
                measured = fit_emission_group(
                    wavelength, flux, stellar_model, components
                )
            except Exception as error:
                measured = [
                    {"line": line, "status": str(error), "detected": False}
                    for line, _ in components
                ]

            for measurement in measured:
                measurement.update({"name": name, "ttype": row["ttype"], "group": group})
                line_rows.append(measurement)

        # Normalize only for spectrum-shape comparison, not line diagnostics.
        try:
            normalization = band_mean(wavelength, flux, NORMALIZATION_BAND)
            if normalization <= 0:
                raise ValueError("Nonpositive normalization continuum.")

            normalized = resample_without_gaps(
                wavelength, flux / normalization, GRID
            )
            np.savez_compressed(
                OUTPUT_DIR / "spectra_cache" / f"{name}.npz",
                wave=GRID,
                normalized_flux=normalized,
            )
            qc["stack_status"] = "cached"
        except Exception as error:
            qc["stack_status"] = str(error)

        feature_rows.append(row)
        qc["status"] = "processed"

    except Exception as error:
        qc["status"] = str(error)

    qc_rows.append(qc)

    # Periodic checkpoints help protect a long Colab session.
    if len(qc_rows) % 25 == 0:
        pd.DataFrame(feature_rows).to_csv(OUTPUT_DIR / "features_checkpoint.csv", index=False)
        pd.DataFrame(line_rows).to_csv(OUTPUT_DIR / "lines_checkpoint.csv", index=False)
        pd.DataFrame(qc_rows).to_csv(OUTPUT_DIR / "qc_checkpoint.csv", index=False)

features = pd.DataFrame(feature_rows)
lines = pd.DataFrame(line_rows)
qc = pd.DataFrame(qc_rows)

if features.empty:
    raise RuntimeError("No galaxies processed successfully; inspect QC.")

features.to_csv(OUTPUT_DIR / "galaxy_features.csv", index=False)
lines.to_csv(OUTPUT_DIR / "emission_lines.csv", index=False)
qc.to_csv(OUTPUT_DIR / "quality_control.csv", index=False)

display(qc["status"].value_counts())
display(qc.head())