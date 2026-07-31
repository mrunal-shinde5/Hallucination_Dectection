import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# FILE
# ============================================================

INPUT_FILE = "../results/triviaqa_10_combined_features.csv"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("WEPR + SEMANTIC FEATURE ANALYSIS")
print("=" * 70)

print("\nLoaded:", len(df), "questions")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# TARGET
# ============================================================

y = df["actual_label"].astype(int)


# ============================================================
# FEATURE GROUPS
# ============================================================

wepr_features = [
    "wepr_hallucination_probability"
]

semantic_features = [
    "semantic_hallucination_score",
    "normalized_entropy"
]


# ============================================================
# CREATE GROUP SCORES
# ============================================================

df["wepr_score"] = (
    df["wepr_hallucination_probability"]
)


# Semantic score:
#
# semantic_hallucination_score is already:
#
# HIGH -> more hallucination
#
# normalized_entropy is also:
#
# HIGH -> more uncertainty
#
# So we average them after normalization.
# ============================================================

scaler = MinMaxScaler()

semantic_matrix = scaler.fit_transform(
    df[
        [
            "semantic_hallucination_score",
            "normalized_entropy"
        ]
    ]
)

df["semantic_score"] = (
    semantic_matrix[:, 0]
    +
    semantic_matrix[:, 1]
) / 2


# ============================================================
# COMBINED SCORE
# ============================================================

# Equal weighting for the initial experiment.
#
# This is NOT trained.
# It is a transparent baseline.

df["combined_score"] = (
    0.5 * df["wepr_score"]
    +
    0.5 * df["semantic_score"]
)


# ============================================================
# DISPLAY SCORES
# ============================================================

print("\n" + "=" * 70)
print("QUESTION-LEVEL SCORES")
print("=" * 70)

display_columns = [
    "id",
    "actual_label",
    "wepr_score",
    "semantic_score",
    "combined_score"
]

print(
    df[display_columns].to_string(
        index=False
    )
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_detector(
    name,
    scores,
    labels
):

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    # --------------------------------------------------------
    # Threshold
    #
    # 0.5 = hallucination
    # --------------------------------------------------------

    predictions = (
        scores >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
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

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    try:

        roc_auc = roc_auc_score(
            labels,
            scores
        )

        print(
            f"ROC-AUC:   {roc_auc:.4f}"
        )

    except ValueError:

        print(
            "ROC-AUC: unavailable"
        )

    # --------------------------------------------------------
    # PR-AUC
    # --------------------------------------------------------

    try:

        pr_auc = average_precision_score(
            labels,
            scores
        )

        print(
            f"PR-AUC:    {pr_auc:.4f}"
        )

    except ValueError:

        print(
            "PR-AUC: unavailable"
        )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        labels,
        predictions
    )

    print("\nConfusion Matrix:")

    print(
        "              Predicted"
    )

    print(
        "              Correct  Hallucination"
    )

    print(
        f"Actual Correct      "
        f"{cm[0,0]:<8}"
        f"{cm[0,1]}"
    )

    print(
        f"Actual Halluc.     "
        f"{cm[1,0]:<8}"
        f"{cm[1,1]}"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc if "roc_auc" in locals() else np.nan,
        "pr_auc": pr_auc if "pr_auc" in locals() else np.nan
    }


# ============================================================
# E1 — WEPR
# ============================================================

wepr_results = evaluate_detector(
    "E1 — WEPR",
    df["wepr_score"],
    y
)


# ============================================================
# E2 — SEMANTIC
# ============================================================

semantic_results = evaluate_detector(
    "E2 — SEMANTIC",
    df["semantic_score"],
    y
)


# ============================================================
# E3 — COMBINED
# ============================================================

combined_results = evaluate_detector(
    "E3 — WEPR + SEMANTIC",
    df["combined_score"],
    y
)


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Experiment": [
        "WEPR",
        "Semantic",
        "WEPR + Semantic"
    ],

    "Accuracy": [
        wepr_results["accuracy"],
        semantic_results["accuracy"],
        combined_results["accuracy"]
    ],

    "Precision": [
        wepr_results["precision"],
        semantic_results["precision"],
        combined_results["precision"]
    ],

    "Recall": [
        wepr_results["recall"],
        semantic_results["recall"],
        combined_results["recall"]
    ],

    "F1": [
        wepr_results["f1"],
        semantic_results["f1"],
        combined_results["f1"]
    ],

    "ROC-AUC": [
        wepr_results["roc_auc"],
        semantic_results["roc_auc"],
        combined_results["roc_auc"]
    ],

    "PR-AUC": [
        wepr_results["pr_auc"],
        semantic_results["pr_auc"],
        combined_results["pr_auc"]
    ]
})


print("\n" + "=" * 70)
print("E1 vs E2 vs E3")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# FEATURE CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE CORRELATION WITH HALLUCINATION LABEL")
print("=" * 70)

correlation_features = [
    "wepr_hallucination_probability",
    "semantic_hallucination_score",
    "normalized_entropy",
    "semantic_consistency",
    "semantic_entropy"
]

correlations = (
    df[
        correlation_features + ["actual_label"]
    ]
    .corr()["actual_label"]
    .drop("actual_label")
    .sort_values(
        ascending=False
    )
)

print(
    correlations
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE = (
    "../results/"
    "triviaqa_10_e3_evaluation.csv"
)

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved comparison to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("E3 ANALYSIS COMPLETE")
print("=" * 70)