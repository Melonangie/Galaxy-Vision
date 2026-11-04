from datetime import datetime, timezone
import re
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

# ------------------------------------------------------------
# 1. Validate the current analysis and load the unfilled template
# ------------------------------------------------------------
required_variables = ["features", "qc", "spectrum_paths", "catalog"]
missing = [name for name in required_variables if name not in globals()]
if missing:
    raise RuntimeError(f"Run the analysis cells first. Missing: {missing}")

template_path = OUTPUT_DIR / "morphology_interpretation_report_template.md"
if not template_path.exists():
    raise FileNotFoundError("Run the report-template cell first.")

F = features.copy()
Q = qc.copy()

if F.empty:
    raise RuntimeError("No measured galaxy features are available.")
if F["name"].duplicated().any() or Q["name"].duplicated().any():
    raise ValueError("Duplicate galaxy rows: resolve them before reporting.")
if not set(F["name"]).issubset(set(Q["name"])):
    raise ValueError("Features and QC do not describe the same attempted galaxies.")

timestamp = datetime.now(timezone.utc)
run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
report = template_path.read_text(encoding="utf-8")


# ------------------------------------------------------------
# 2. Small Markdown helpers; no extra package is necessary
# ------------------------------------------------------------
def markdown_table(table):
    """Convert a DataFrame to readable Markdown without tabulate."""
    if table is None or table.empty:
        return "_Unavailable or not calculated in this run._"

    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4g}"
        return str(value).replace("|", r"\|").replace("\n", " ")

    header = "| " + " | ".join(map(str, table.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def replace_first_table(text, heading, table):
    """Replace the placeholder table only within the named subsection."""
    start = text.find(heading)
    if start == -1:
        return text

    # Stop at the next heading of equal or higher level.
    level = len(heading) - len(heading.lstrip("#"))
    next_heading = re.search(
        rf"^#{{1,{level}}} ", text[start + len(heading):], re.MULTILINE
    )
    end = (
        start + len(heading) + next_heading.start()
        if next_heading else len(text)
    )
    section = text[start:end]
    section = re.sub(
        r"^\|.*(?:\n\|.*)*",
        lambda _: markdown_table(table),
        section, count=1, flags=re.MULTILINE,
    )
    return text[:start] + section + text[end:]


def optional_table(variable, columns):
    """Only use optional results that exist as nonempty DataFrames."""
    table = globals().get(variable)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return pd.DataFrame()
    return table[[c for c in columns if c in table.columns]].copy()


# ------------------------------------------------------------
# 3. Counts: distinguish full inventory from the attempted subset
# ------------------------------------------------------------
inventory = pd.DataFrame({
    "name": [galaxy_id(path) for path in spectrum_paths]
})

# Counts refer to unique galaxies, not numbers of FITS files.
duplicate_files = int(inventory["name"].duplicated().sum())
inventory = inventory.drop_duplicates("name")

catalog_labels = catalog.reset_index()[["name", "ttype"]].copy()
if "type" in catalog.columns:
    catalog_labels["type"] = catalog.reset_index()["type"]

inventory = inventory.merge(
    catalog_labels, on="name", how="left", validate="one_to_one",
    indicator=True
)

attempted_names = set(Q["name"])
processed_names = set(F["name"])

if "starlight_status" in Q:
    aligned_names = set(
        Q.loc[Q["starlight_status"].eq("aligned"), "name"]
    )
else:
    aligned_names = set()

if "stack_status" in Q:
    stacked_names = set(Q.loc[Q["stack_status"].eq("cached"), "name"])
else:
    stacked_names = set()

inventory["attempted"] = inventory["name"].isin(attempted_names)
inventory["processed"] = inventory["name"].isin(processed_names)
inventory["aligned"] = inventory["name"].isin(aligned_names)
inventory["stacked"] = inventory["name"].isin(stacked_names)

counts = pd.DataFrame([
    ("Unique galaxies discovered", len(inventory)),
    ("Duplicate spectrum files beyond unique IDs", duplicate_files),
    ("Discovered galaxies without catalog match",
     int(inventory["_merge"].eq("left_only").sum())),
    ("Galaxies attempted in this run", len(Q)),
    ("Discovered galaxies not attempted",
     len(set(inventory["name"]) - attempted_names)),
    ("Galaxies with processed rest-frame spectra", len(F)),
    ("Attempted galaxies without processed spectra",
     len(attempted_names - processed_names)),
    ("Galaxies with validated STARLIGHT continua", len(aligned_names)),
    ("Galaxies successfully normalized for stacking", len(stacked_names)),
    ("Processed galaxies missing Ttype", int(F["ttype"].isna().sum())),
], columns=["Stage / outcome", "Count"])

# Build the morphology table from discovered galaxies, not only successes.
morphology = (
    inventory.groupby("ttype", dropna=False)
    .agg(
        discovered=("name", "size"),
        attempted=("attempted", "sum"),
        processed=("processed", "sum"),
        starlight_aligned=("aligned", "sum"),
        normalized_for_stacking=("stacked", "sum"),
    )
    .reset_index()
)

if "type" in inventory:
    type_strings = (
        inventory.groupby("ttype", dropna=False)["type"]
        .agg(lambda values: ", ".join(sorted(set(values.dropna().astype(str)))))
        .reset_index(name="Catalog Type strings")
    )
    morphology = morphology.merge(type_strings, on="ttype", how="left")

morphology["Label caveat"] = np.where(
    morphology["ttype"].eq(11),
    "Verify catalog convention; do not assume irregular.",
    "Catalog labels retained; uncertainty not quantified."
)

report = report.replace(
    "> Status: TEMPLATE — no scientific findings are implied.",
    "> Status: AUTO-PREFILLED DRAFT — measured tables inserted; "
    "interpretation and validation remain incomplete."
)
report = report.replace(
    "- Galaxies discovered: [N]",
    f"- Galaxies discovered: {len(inventory)} unique galaxies"
)
report = report.replace(
    "- Galaxies with usable rest-frame spectra: [N]",
    f"- Galaxies with usable rest-frame spectra: {len(F)} processed galaxies "
    f"out of {len(Q)} attempted"
)
report = report.replace(
    "- Galaxies with validated STARLIGHT continua: [N]",
    f"- Galaxies with validated STARLIGHT continua: {len(aligned_names)}"
)
report = report.replace(
    "- Morphological stages represented: [list catalog Ttypes and Type strings]",
    "- Morphological stages represented in processed spectra: "
    + ", ".join(map(str, sorted(F["ttype"].dropna().unique())))
)
report = report.replace(
    "- Run date / analysis identifier: [...]",
    f"- Report generation time / identifier: {timestamp.isoformat()} / {run_id}"
)

report = replace_first_table(report, "### 3.1 Catalog conventions", morphology)
report = replace_first_table(report, "### 3.2 Sample flow and exclusions", counts)


# ------------------------------------------------------------
# 4. Populate measured overall and pairwise feature rankings
# ------------------------------------------------------------
overall_results = optional_table(
    "overall", ["feature", "n", "types", "epsilon_squared", "p", "q_fdr"]
)
pairwise_results = optional_table(
    "pairwise",
    ["feature", "ttype_a", "ttype_b", "n_a", "n_b",
     "rank_biserial", "absolute_effect", "q_fdr"]
)

if not overall_results.empty:
    overall_results = overall_results.sort_values(
        "epsilon_squared", ascending=False
    )
    overall_results["Coverage caveat"] = (
        "Available-case analysis; compare availability by Ttype."
    )
    report = replace_first_table(
        report, "### 9.1 Overall associations", overall_results.head(20)
    )
    report = report.replace(
        "- Most informative features: [results, not prior expectations]",
        "- Largest measured overall effect sizes: "
        + ", ".join(overall_results["feature"].head(5))
        + " (association ranking, not proof of predictive importance)"
    )

if not pairwise_results.empty:
    pairwise_results = pairwise_results.sort_values(
        "absolute_effect", ascending=False
    )
    pairwise_results["Bootstrap interval"] = "Not calculated by this workflow."
    report = replace_first_table(
        report, "### 9.2 Pairwise separation", pairwise_results.head(20)
    )

    strongest = pairwise_results.iloc[0]
    report = report.replace(
        "- Best-separated type pairs: [pairs, effect sizes, uncertainty]",
        f"- Largest measured pair-feature effect: "
        f"T={strongest['ttype_a']} versus T={strongest['ttype_b']}, "
        f"{strongest['feature']}, "
        f"|rank-biserial|={strongest['absolute_effect']:.3f}; "
        "bootstrap uncertainty not calculated"
    )

# Do not automatically infer the most overlapping pair from a weak
# univariate effect: multivariate overlap requires separate evidence.


# ------------------------------------------------------------
# 5. BPT morphology tables with explicit denominators
# ------------------------------------------------------------
bpt_sections = []
for panel in ["NII", "SII", "OI"]:
    column = f"bpt_{panel}_class"
    if column not in F:
        continue

    data = F[["ttype", column]].copy()
    data[column] = data[column].fillna("unclassified")

    table = pd.crosstab(data["ttype"], data[column]).reset_index()
    category_columns = [c for c in table if c != "ttype"]
    table["N processed with Ttype"] = table[category_columns].sum(axis=1)
    table["N classifiable"] = (
        table["N processed with Ttype"] - table.get("unclassified", 0)
    )
    table["Classifiable fraction"] = (
        table["N classifiable"] / table["N processed with Ttype"]
    )

    bpt_sections.append(
        f"#### Measured {panel} BPT counts\n\n"
        + markdown_table(table)
        + "\n\nDenominator: processed galaxies with a catalog Ttype. "
        "Unclassified includes unavailable required inputs.\n"
    )

if bpt_sections:
    # Remove the generic placeholder table, then insert each panel separately.
    report = replace_first_table(
        report, "### 7.2 BPT results", pd.DataFrame()
    )
    marker = "### 7.2 BPT results"
    report = report.replace(
        marker, marker + "\n\n" + "\n".join(bpt_sections), 1
    )


# ------------------------------------------------------------
# 6. Optional held-out classification results
# ------------------------------------------------------------
cv_scores = optional_table(
    "score_table", ["fold", "balanced_accuracy", "macro_f1"]
)
importance_ranking = globals().get("ranking")

classification_sections = []
if not cv_scores.empty:
    classification_sections.append(
        "### Measured cross-validation scores\n\n" + markdown_table(cv_scores)
    )
    for metric, placeholder in [
        ("balanced_accuracy",
         "- Cross-validated balanced accuracy: [mean and fold variation]"),
        ("macro_f1",
         "- Cross-validated macro F1: [mean and fold variation]"),
    ]:
        if metric in cv_scores:
            mean = cv_scores[metric].mean()
            std = cv_scores[metric].std()
            report = report.replace(
                placeholder,
                f"- Cross-validated {metric}: {mean:.3f}; "
                f"across-fold standard deviation {std:.3f}"
            )

if isinstance(importance_ranking, pd.DataFrame) and not importance_ranking.empty:
    importance_display = importance_ranking.reset_index()
    classification_sections.append(
        "### Measured held-out permutation-importance ranking\n\n"
        + markdown_table(importance_display.head(20))
        + "\n\nImportance is the change in the configured validation score "
        "after permutation. Negative values do not establish harmful physical "
        "effects. Fold variation is not a confidence interval."
    )

# Report actual failure categories without guessing their physical causes.
qc_summary = (
    Q["status"].fillna("missing status").value_counts()
    .rename_axis("QC outcome").reset_index(name="N attempted")
)

report += (
    "\n\n## Automatically generated run details\n\n"
    "This draft uses current notebook variables. Tables show up to 20 ranked "
    "entries; complete overall/pairwise rankings are saved alongside the report. "
    "Optional classification outputs are included only when present; verify "
    "they were calculated for this same run.\n\n"
    "### Actual processing outcomes\n\n"
    + markdown_table(qc_summary)
    + "\n\n"
    + ("\n\n".join(classification_sections)
       if classification_sections else
       "_No optional classification results available._")
    + "\n"
)


# ------------------------------------------------------------
# 7. Save a new draft and supporting tables; preserve edited reports
# ------------------------------------------------------------
report_path = OUTPUT_DIR / f"morphology_report_prefilled_{run_id}.md"
report_path.write_text(report, encoding="utf-8")

counts.to_csv(OUTPUT_DIR / f"report_counts_{run_id}.csv", index=False)
morphology.to_csv(OUTPUT_DIR / f"report_morphology_{run_id}.csv", index=False)

for label, table in [
    ("overall_rankings", overall_results),
    ("pairwise_rankings", pairwise_results),
]:
    if not table.empty:
        table.to_csv(OUTPUT_DIR / f"report_{label}_{run_id}.csv", index=False)

print("Saved report:", report_path)
print("Interpretation prompts remain intentionally unfilled.")
display(Markdown(report[:report.find("## 2.")]))