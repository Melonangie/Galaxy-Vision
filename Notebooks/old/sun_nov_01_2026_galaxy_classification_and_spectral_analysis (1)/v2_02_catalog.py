cat = Table.read(MORPH_FILE).to_pandas()

# FITS text columns sometimes come as bytes (b'manga-...'): turn them into normal strings
for col in ["name", "Type"]:
    cat[col] = cat[col].apply(lambda s: s.decode() if isinstance(s, bytes) else s).str.strip()

cat = cat[cat["nsa_z"] > 0]                      # -9999 = no redshift -> cannot de-redshift
cat = cat[cat["Ttype"].between(-5, 10)].copy()   # 11 = 'S' (no subtype) is not on the scale
cat["Ttype"] = cat["Ttype"].astype(int)

# Bar information from the Type text: SB = strong bar, SAB = weak bar
cat["bar"] = np.select([cat.Type.str.startswith("SB"), cat.Type.str.startswith("SAB")],
                       ["strong", "weak"], default="none")
cat = cat.set_index("name")

# One colour per T-type: red (early, old, red galaxies) -> blue (late, young, blue galaxies)
TTYPES = sorted(cat["Ttype"].unique())
TT_COLORS = dict(zip(TTYPES, sns.color_palette("RdYlBu", len(TTYPES))))
TT_NAMES = {-5:"E", -4:"E+", -3:"S0-", -2:"S0", -1:"S0+", 0:"S0a", 1:"Sa", 2:"Sab", 3:"Sb",
            4:"Sbc", 5:"Sc", 6:"Scd", 7:"Sd", 8:"Sdm", 9:"Sm", 10:"Irr"}
print(cat["Ttype"].value_counts().sort_index())