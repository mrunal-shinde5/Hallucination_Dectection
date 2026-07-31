import pandas as pd
import os


# ============================================================
# FILE PATHS
# ============================================================

WEPR_FILE = "../results/triviaqa_10_results.csv"

SEMANTIC_FILE = "../results/triviaqa_10_semantic_results_labelled.csv"

OUTPUT_FILE = "../results/triviaqa_10_combined_features.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING EXPERIMENT RESULTS")
print("=" * 70)

wepr = pd.read_csv(WEPR_FILE)

semantic = pd.read_csv(SEMANTIC_FILE)

print("\nWEPR shape:", wepr.shape)

print("Semantic shape:", semantic.shape)


# ============================================================
# CHECK IDS
# ============================================================

print("\nChecking IDs...")

wepr_ids = set(wepr["id"])

semantic_ids = set(semantic["id"])

if wepr_ids != semantic_ids:

    print("\nWARNING: IDs do not match.")

    print(
        "WEPR only:",
        sorted(wepr_ids - semantic_ids)
    )

    print(
        "Semantic only:",
        sorted(semantic_ids - wepr_ids)
    )

    raise ValueError(
        "WEPR and semantic datasets do not contain the same IDs."
    )

print("IDs match successfully.")


# ============================================================
# CHECK LABELS
# ============================================================

print("\nChecking ground-truth labels...")

wepr_reference = (
    wepr[
        ["id", "reference_answer"]
    ]
    .sort_values("id")
    .reset_index(drop=True)
)

semantic_reference = (
    semantic[
        ["id", "reference_answer"]
    ]
    .sort_values("id")
    .reset_index(drop=True)
)

if not wepr_reference.equals(
    semantic_reference
):

    print(
        "WARNING: reference-answer fields differ."
    )

    # We don't immediately stop because the actual
    # target label comes from the semantic file.

else:

    print(
        "Reference answers match."
    )


# ============================================================
# SELECT WEPR FEATURES
# ============================================================

wepr_features = wepr[
    [
        "id",
        "non_hallucination_probability",
        "hallucination_probability"
    ]
].copy()


wepr_features = wepr_features.rename(
    columns={
        "non_hallucination_probability":
            "wepr_non_hallucination_probability",

        "hallucination_probability":
            "wepr_hallucination_probability"
    }
)


# ============================================================
# SELECT SEMANTIC FEATURES
# ============================================================

semantic_features = semantic[
    [
        "id",
        "semantic_entropy",
        "normalized_entropy",
        "semantic_consistency",
        "semantic_hallucination_score",
        "actual_label"
    ]
].copy()


# ============================================================
# MERGE
# ============================================================

print("\nMerging WEPR + semantic features...")

combined = pd.merge(
    wepr_features,
    semantic_features,
    on="id",
    how="inner"
)


# ============================================================
# CHECK MERGE
# ============================================================

print(
    "\nCombined dataset shape:",
    combined.shape
)

expected_rows = len(
    semantic_features
)

if len(combined) != expected_rows:

    raise ValueError(
        "Merge lost rows."
    )


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    combined.isnull().sum()
)


# ============================================================
# CHECK LABEL DISTRIBUTION
# ============================================================

print(
    "\nActual label distribution:"
)

print(
    combined["actual_label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

combined.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)

print(
    "COMBINED FEATURE DATASET"
)

print("=" * 70)

print(
    combined.to_string(
        index=False
    )
)

print("\nSaved to:")

print(
    OUTPUT_FILE
)