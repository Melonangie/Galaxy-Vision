# Read the supporting catalogue (FITS table) and keep only what we need
cat = Table.read(MORPH_FILE).to_pandas()

# FITS strings sometimes arrive as bytes (b'manga-...'): convert to normal text
for col in ["name", "Type"]:
    cat[col] = cat[col].apply(lambda s: s.decode() if isinstance(s, bytes) else s).str.strip()

# nsa_z = -9999 means "no redshift" -> we cannot go to the rest frame, drop them
cat = cat[cat["nsa_z"] > 0].copy()
# Ttype 11 = "S" (spiral with unknown subtype) is not on the -5..10 scale -> drop it
cat = cat[cat["Ttype"].between(-5, 10)].copy()

# Group the T-types into broader classes so every class has enough galaxies.
# The boundaries follow the usual early-type / early-spiral / late-spiral split
# (e.g. Nair & Abraham 2010, ApJS 186, 427).
def morph_class(t):
    if t <= -3: return "E"
    if t <= -1: return "S0"
    if t <=  2: return "S0a-Sab"
    if t <=  4: return "Sb-Sbc"
    if t <=  7: return "Sc-Sd"
    return "Sdm-Irr"

CLASS_ORDER = ["E", "S0", "S0a-Sab", "Sb-Sbc", "Sc-Sd", "Sdm-Irr"]
cat["morph_class"] = cat["Ttype"].apply(morph_class)

# Extra information we can get from the Type text: is there a bar? (SB = strong bar, SAB = weak bar)
cat["bar"] = np.select([cat["Type"].str.startswith("SB"), cat["Type"].str.startswith("SAB")],
                       ["strong", "weak"], default="none")

# Colours from red (early type) to blue (late type), to match the real galaxy colours
CLASS_COLORS = dict(zip(CLASS_ORDER, sns.color_palette("RdYlBu", len(CLASS_ORDER))))
cat = cat.set_index("name")
print(cat["morph_class"].value_counts().reindex(CLASS_ORDER))