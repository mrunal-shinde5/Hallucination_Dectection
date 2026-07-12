import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

SEMANTIC_FILE = "../results/triviaqa_10_semantic_results.csv"

# This is your E1 labelled file.
# It contains the deterministic Qwen answer and actual_label.
E1_FILE = "../results/triviaqa_100_results_labelled.csv"

OUTPUT_FILE = (
    "../results/"
    "triviaqa_10_semantic_results_labelled.csv"
)

THRESHOLD = 0.50


# ============================================================
# LOAD E2 RESULTS
# ============================================================

semantic_df = pd.read_csv(
    SEMANTIC_FILE
)

print(
    f"Loaded {len(semantic_df)} E2 semantic results."
)


# ============================================================
# LOAD E1 RESULTS
# ============================================================

e1_df = pd.read_csv(
    E1_FILE
)

print(
    f"Loaded {len(e1_df)} E1 results."
)


# ============================================================
# SELECT ONLY INFORMATION NEEDED FROM E1
# ============================================================

e1_labels = e1_df[
    [
        "id",
        "question",
        "reference_answer",
        "generated_answer",
        "actual_label"
    ]
].copy()


# Rename deterministic answer so it is obvious
# that this came from E1.

e1_labels = e1_labels.rename(
    columns={
        "generated_answer":
            "primary_deterministic_answer"
    }
)


# ============================================================
# MERGE E1 LABELS WITH E2
# ============================================================

df = semantic_df.merge(
    e1_labels,
    on="id",
    how="left",
    suffixes=("_semantic", "_e1")
)


# ============================================================
# CHECK THAT EVERY QUESTION MATCHED
# ============================================================

missing_labels = df[
    df["actual_label"].isna()
]

if len(missing_labels) > 0:

    print(
        "\nERROR: Some E2 questions do not have "
        "an E1 ground-truth label."
    )

    print(
        missing_labels[
            ["id", "question_semantic"]
        ]
    )

    raise ValueError(
        "Missing E1 labels."
    )


# ============================================================
# USE PRIMARY/DETERMINISTIC LABEL
# ============================================================

# 0 = primary answer correct
# 1 = primary answer incorrect/hallucinated

df["actual_label"] = (
    df["actual_label"]
    .astype(int)
)


# ============================================================
# CREATE SEMANTIC HALLUCINATION SCORE
# ============================================================

# Semantic consistency:
#
#   HIGH = generations agree
#   LOW  = generations disagree
#
# Therefore:
#
#   HIGH semantic hallucination score
#       = high disagreement
#
# We invert the consistency score.

df["semantic_hallucination_score"] = (
    1.0 -
    df["semantic_consistency"]
)


# ============================================================
# PREDICTION
# ============================================================

df["predicted_label"] = (
    df["semantic_hallucination_score"]
    >= THRESHOLD
).astype(int)


# ============================================================
# PREPARE EVALUATION DATA
# ============================================================

valid_df = df[
    df["semantic_hallucination_score"]
    .notna()
].copy()


y_true = valid_df[
    "actual_label"
]

y_score = valid_df[
    "semantic_hallucination_score"
]

y_pred = valid_df[
    "predicted_label"
]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

roc_auc = roc_auc_score(
    y_true,
    y_score
)

pr_auc = average_precision_score(
    y_true,
    y_score
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("E2 — SEMANTIC CONSISTENCY")
print("=" * 70)

print(
    "\nGround truth is based ONLY on the "
    "primary/deterministic E1 answer."
)

print(
    "\nCorrect primary answers:",
    (y_true == 0).sum()
)

print(
    "Incorrect primary answers:",
    (y_true == 1).sum()
)


print("\nQuestion-level results:")

display_columns = [
    "id",
    "primary_deterministic_answer",
    "reference_answer",
    "actual_label",
    "semantic_consistency",
    "semantic_hallucination_score",
    "predicted_label"
]

print(
    valid_df[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# METRICS
# ============================================================

print("\n" + "=" * 70)
print("SEMANTIC CONSISTENCY METRICS")
print("=" * 70)

print(
    f"\nAccuracy:  {accuracy:.4f}"
)

print(
    f"ROC-AUC:   {roc_auc:.4f}"
)

print(
    f"PR-AUC:    {pr_auc:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1-score:  {f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    "\n                 Predicted"
)

print(
    "                 Correct  Hallucination"
)

print(
    f"Actual Correct      "
    f"{cm[0][0]:<8}"
    f"{cm[0][1]}"
)

print(
    f"Actual Halluc.     "
    f"{cm[1][0]:<8}"
    f"{cm[1][1]}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nSaved labelled E2 results to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# IMPORTANT INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    """
The semantic score does NOT determine whether
the answer is factually correct.

The actual_label comes from the primary,
deterministic Qwen answer used in E1.

The five stochastic answers are used only to
measure semantic consistency.
"""
)