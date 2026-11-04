features_rows = []
failures = []
cache_dir = OUTPUT_DIR / "normalized_spectra"
cache_dir.mkdir(exist_ok=True)

for item in tqdm(usable.to_dict("records")):
    identity = item["galaxy_id"]
    info = catalog.loc[identity]

    try:
        z = float(info["nsa_z"])
        wave, flux = load_spectrum(item["spectrum_path"], z)

        row = {
            "galaxy_id": identity,
            "ttype": info["ttype"],
            "type": info["type"],
            "nsa_z": z,
            **stellar_features(wave, flux),
            "continuum_trusted": False,
        }

        # Supporting properties are context, not morphology-derived predictors.
        for name in ["logmass", "incl", "petro_th90"]:
            row[name] = info.get(name, np.nan)

        if item["starlight_path"]:
            try:
                synthetic, normalization, population = read_starlight(
                    item["starlight_path"]
                )
                row.update(population)
                continuum, trusted, mismatch = validated_continuum(
                    wave, flux, synthetic, normalization, z
                )
                row["continuum_trusted"] = trusted
                row["starlight_flux_mismatch"] = mismatch

                if trusted and WAVELENGTH_CONVENTION == "vacuum":
                    residual = flux - continuum
                    for group_name, group in LINE_GROUPS.items():
                        measurements, status = fit_emission_group(
                            wave, residual, continuum, group
                        )
                        row.update(measurements)
                        row[f"{group_name}_fit_status"] = status
                else:
                    row["line_measurement_status"] = "untrusted_continuum_or_convention"
            except Exception as error:
                row["starlight_failure"] = str(error)

        normalization = band_mean(wave, flux, NORMALIZATION_BAND)
        row["normalization_valid"] = bool(
            np.isfinite(normalization) and normalization > 0
        )

        if row["normalization_valid"]:
            normalized = regrid_without_gaps(wave, flux / normalization, REST_GRID)
            np.savez_compressed(
                cache_dir / f"{identity}.npz",
                wave=REST_GRID, flux=normalized,
            )

        features_rows.append(row)

    except Exception as error:
        failures.append({"galaxy_id": identity, "error": str(error)})

    # Checkpoint the measurement table, but do not silently reuse stale caches.
    pd.DataFrame(features_rows).to_csv(
        OUTPUT_DIR / "galaxy_features.csv", index=False
    )

df = pd.DataFrame(features_rows)
if df.empty:
    raise ValueError("No galaxies processed successfully; inspect failures.")

pd.DataFrame(failures, columns=["galaxy_id", "error"]).to_csv(
    OUTPUT_DIR / "failures.csv", index=False
)

print(f"Measured {len(df)} galaxies; {len(failures)} failed.")