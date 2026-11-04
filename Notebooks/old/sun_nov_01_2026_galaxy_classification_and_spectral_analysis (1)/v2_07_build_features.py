sl_lookup = {galaxy_name(p): p for p in out_files}
rows = []
for i, name in enumerate(tqdm(spec_index, desc="features")):
    row = {"name": name}
    model_on_grid = None
    if name in sl_lookup:
        try:
            info, wl, fo, fs = read_starlight(sl_lookup[name])
            row.update(info)
            row.update(measure_emission(wl, fo, fs))
            model_on_grid = np.interp(GRID, wl, fs)          # stellar model for Ca II H/K
        except Exception as e:
            print("STARLIGHT problem:", name, e)
    row.update(absorption_features(GRID, SPEC[i], model_on_grid))
    rows.append(row)

feat = pd.DataFrame(rows).set_index("name").join(cat, how="inner")

# -------- derived diagnostics (only when all lines have S/N >= 3) --------
ok = lambda *ls: np.all([feat[f"SN_{l}"] >= 3 for l in ls], axis=0)
safe_log = lambda a, b, cond: np.where(cond, np.log10(a / b), np.nan)

feat["log_NII_Ha"]   = safe_log(feat.F_NII6584, feat.F_Ha, ok("NII6584", "Ha"))
feat["log_OIII_Hb"]  = safe_log(feat.F_OIII5007, feat.F_Hb, ok("OIII5007", "Hb"))
feat["log_SII_Ha"]   = safe_log(feat.F_SII6717 + feat.F_SII6731, feat.F_Ha, ok("SII6717", "SII6731", "Ha"))
feat["log_OI_Ha"]    = safe_log(feat.F_OI6300, feat.F_Ha, ok("OI6300", "Ha"))
feat["log_NeIII_OII"] = safe_log(feat.F_NeIII3869, feat.F_OII3727, ok("NeIII3869", "OII3727"))  # ionisation
feat["log_ArIII_OIII"] = safe_log(feat.F_ArIII7136, feat.F_OIII5007, ok("ArIII7136", "OIII5007"))
feat["OIII_4363_5007"] = np.where(ok("OIII4363", "OIII5007"), feat.F_OIII4363 / feat.F_OIII5007, np.nan)  # T_e
feat["SII_6717_6731"]  = np.where(ok("SII6717", "SII6731"), feat.F_SII6717 / feat.F_SII6731, np.nan)     # n_e
feat["log_EW_Ha"]      = np.log10(feat.EW_Ha.clip(lower=0.01))

# Balmer decrements vs Case B, T = 10^4 K (Osterbrock & Ferland 2006):
#   Ha/Hb = 2.86, Hg/Hb = 0.468, Hd/Hb = 0.259. Dust raises Ha/Hb and lowers Hg/Hb, Hd/Hb.
feat["Ha_Hb"] = np.where(ok("Ha", "Hb"), feat.F_Ha / feat.F_Hb, np.nan)
feat["Hg_Hb"] = np.where(ok("Hg", "Hb"), feat.F_Hg / feat.F_Hb, np.nan)
feat["Hd_Hb"] = np.where(ok("Hd", "Hb"), feat.F_Hd / feat.F_Hb, np.nan)
# Gas extinction with the CCM law: k(Hb)=3.61, k(Ha)=2.53, R_V=3.1
feat["AV_gas"] = 3.1 * 2.5 / (3.61 - 2.53) * np.log10(np.clip(feat.Ha_Hb, 2.86, None) / 2.86)

# Good STARLIGHT fits only (adjust the thresholds after looking at Cell 8)
feat["good_fit"] = (feat.sl_adev < 10) & (feat.sl_SN > 5)
feat.to_csv(f"{SAVE_DIR}/features.csv")
print(feat.shape)