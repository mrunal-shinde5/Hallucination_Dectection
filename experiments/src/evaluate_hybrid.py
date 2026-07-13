import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

INPUT_FILE = "../results/triviaqa_100_E1_E2_combined.csv"

OUTPUT_FILE = "../results/triviaqa_100_hybrid_results.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(
    f"Loaded {len(df)} questions."
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "id",
    "actual_label",
    "hallucination_probability",
    "semantic_hallucination_score"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )


# ============================================================
# REMOVE ROWS WITH MISSING SCORES
# ============================================================

df = df.dropna(
    subset=[
        "hallucination_probability",
        "semantic_hallucination_score",
        "actual_label"
    ]
).copy()


print(
    f"Questions available for hybrid experiment: "
    f"{len(df)}"
)


# ============================================================
# FEATURES
# ============================================================

# Feature 1:
# WEPR hallucination probability

# Feature 2:
# Semantic hallucination score
#
# High semantic hallucination score means
# low semantic consistency.

X = df[
    [
        "hallucination_probability",
        "semantic_hallucination_score"
    ]
]


y = df[
    "actual_label"
].astype(int)


# ============================================================
# HYBRID MODEL
# ============================================================

# StandardScaler:
# Puts both features on a comparable scale.
#
# LogisticRegression:
# Learns how the two signals jointly relate
# to hallucination.

hybrid_model = Pipeline(
    steps=[

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )

    ]
)


# ============================================================
# STRATIFIED 5-FOLD CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print(
    "\nRunning 5-fold cross-validation..."
)


# cross_val_predict gives each question a prediction
# from a model that did NOT train on that question.

hybrid_probability = cross_val_predict(
    hybrid_model,
    X,
    y,
    cv=cv,
    method="predict_proba"
)[:, 1]


hybrid_prediction = (
    hybrid_probability >= 0.5
).astype(int)


# ============================================================
# HYBRID METRICS
# ============================================================

accuracy = accuracy_score(
    y,
    hybrid_prediction
)

roc_auc = roc_auc_score(
    y,
    hybrid_probability
)

pr_auc = average_precision_score(
    y,
    hybrid_probability
)

precision = precision_score(
    y,
    hybrid_prediction,
    zero_division=0
)

recall = recall_score(
    y,
    hybrid_prediction,
    zero_division=0
)

f1 = f1_score(
    y,
    hybrid_prediction,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y,
    hybrid_prediction,
    labels=[0, 1]
)


# ============================================================
# SAVE HYBRID PREDICTIONS
# ============================================================

df[
    "hybrid_hallucination_probability"
] = hybrid_probability


df[
    "hybrid_predicted_label"
] = hybrid_prediction


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY HYBRID RESULTS
# ============================================================

print("\n" + "=" * 70)

print(
    "E3 — HYBRID WEPR + SEMANTIC CONSISTENCY"
)

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

print(
    "\nConfusion Matrix:"
)

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
# DISPLAY INDIVIDUAL PREDICTIONS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "QUESTION-LEVEL HYBRID RESULTS"
)

print(
    "=" * 70
)


display_columns = [

    "id",

    "actual_label",

    "hallucination_probability",

    "semantic_hallucination_score",

    "hybrid_hallucination_probability",

    "hybrid_predicted_label"

]


print(
    df[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# SAVE LOCATION
# ============================================================

print(
    "\nHybrid results saved to:"
)

print(
    OUTPUT_FILE
)