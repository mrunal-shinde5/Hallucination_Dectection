import pandas as pd
import numpy as np

from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from sklearn.base import clone

import os


# ============================================================
# CONFIGURATION
# ============================================================

WEPR_FILE = "../results/triviaqa_500_wepr_results.csv"

SEMANTIC_FILE = (
    "../results/triviaqa_500_semantic_results_labelled.csv"
)

E3_FILE = (
    "../results/triviaqa_500_E3_features.csv"
)

OUTPUT_DIR = "../results/final_500"

RANDOM_STATE = 42

N_SPLITS = 5


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("FINAL 500-QUESTION EVALUATION")
print("E1 vs E2 vs E3")
print("=" * 80)


wepr_df = pd.read_csv(
    WEPR_FILE
)

semantic_df = pd.read_csv(
    SEMANTIC_FILE
)

e3_df = pd.read_csv(
    E3_FILE
)


print(
    f"\nWEPR rows:     {len(wepr_df)}"
)

print(
    f"Semantic rows: {len(semantic_df)}"
)

print(
    f"E3 rows:       {len(e3_df)}"
)


# ============================================================
# VALIDATE DATASET SIZE
# ============================================================

for name, data in [

    ("WEPR", wepr_df),

    ("Semantic", semantic_df),

    ("E3", e3_df)

]:

    if len(data) != 500:

        raise ValueError(

            f"{name} dataset contains "
            f"{len(data)} rows instead of 500."

        )


# ============================================================
# CHECK REQUIRED WEPR COLUMN
# ============================================================

if "hallucination_probability" not in wepr_df.columns:

    raise ValueError(
        "WEPR file does not contain "
        "'hallucination_probability'."
    )


# ============================================================
# CHECK REQUIRED SEMANTIC COLUMNS
# ============================================================

semantic_required_columns = [

    "id",

    "semantic_entropy",

    "semantic_consistency",

    "answer_1",

    "answer_2",

    "answer_3",

    "answer_4",

    "answer_5"

]


missing_semantic = [

    column

    for column in semantic_required_columns

    if column not in semantic_df.columns

]


if missing_semantic:

    raise ValueError(

        "Semantic file is missing columns:\n"

        + "\n".join(
            missing_semantic
        )

    )


# ============================================================
# CHECK REQUIRED E3 COLUMNS
# ============================================================

e3_required_columns = [

    "id",

    "actual_label",

    "hallucination_probability",

    "semantic_consistency",

    "semantic_entropy",

    "unique_answer_count",

    "majority_agreement"

]


missing_e3 = [

    column

    for column in e3_required_columns

    if column not in e3_df.columns

]


if missing_e3:

    raise ValueError(

        "E3 file is missing columns:\n"

        + "\n".join(
            missing_e3
        )

    )


# ============================================================
# CHECK IDS
# ============================================================

wepr_ids = set(
    wepr_df["id"]
)

semantic_ids = set(
    semantic_df["id"]
)

e3_ids = set(
    e3_df["id"]
)


if wepr_ids != semantic_ids:

    raise ValueError(
        "WEPR and semantic IDs do not match."
    )


if wepr_ids != e3_ids:

    raise ValueError(
        "WEPR and E3 IDs do not match."
    )


print(
    "\nID consistency check: PASSED"
)


# ============================================================
# TARGET
# ============================================================

y = e3_df[
    "actual_label"
].astype(int)


print(
    "\n"
    + "=" * 70
)

print(
    "LABEL DISTRIBUTION"
)

print(
    "=" * 70
)

print(
    y.value_counts()
    .sort_index()
)


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(

    n_splits=N_SPLITS,

    shuffle=True,

    random_state=RANDOM_STATE

)


# ============================================================
# STORAGE
# ============================================================

all_results = []

oof_predictions = {}

oof_probabilities = {}

confusion_matrices = {}


# ============================================================
# GENERIC EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    X,
    y,
    model
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Evaluating: {name}"
    )

    print(
        "=" * 70
    )


    probabilities = np.zeros(
        len(y)
    )

    predictions = np.zeros(
        len(y),
        dtype=int
    )


    # --------------------------------------------------------
    # 5-FOLD OUT-OF-FOLD EVALUATION
    # --------------------------------------------------------

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(

        cv.split(
            X,
            y
        ),

        start=1

    ):

        print(
            f"Fold {fold}/{N_SPLITS}"
        )


        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]


        fold_model = clone(
            model
        )


        fold_model.fit(
            X_train,
            y_train
        )


        fold_probabilities = (

            fold_model

            .predict_proba(
                X_test
            )[:, 1]

        )


        fold_predictions = (

            fold_probabilities >= 0.5

        ).astype(int)


        probabilities[
            test_idx
        ] = fold_probabilities


        predictions[
            test_idx
        ] = fold_predictions


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    oof_probabilities[
        name
    ] = probabilities

    oof_predictions[
        name
    ] = predictions

    confusion_matrices[
        name
    ] = cm


    all_results.append({

        "Method":
            name,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "ROC-AUC":
            roc_auc,

        "PR-AUC":
            pr_auc

    })


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC   : {pr_auc:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        cm
    )


# ============================================================
# E1 — WEPR
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "E1 — WEPR"
)

print(
    "=" * 80
)


X_wepr = wepr_df[
    [
        "hallucination_probability"
    ]
].astype(float)


# Make sure WEPR rows are in the same ID order as E3

wepr_ordered = (
    wepr_df
    .set_index("id")
    .loc[e3_df["id"]]
    .reset_index()
)


X_wepr = wepr_ordered[
    [
        "hallucination_probability"
    ]
].astype(float)


evaluate_model(

    "E1 — WEPR",

    X_wepr,

    y,

    Pipeline([

        (
            "scaler",

            StandardScaler()

        ),

        (
            "classifier",

            LogisticRegression(

                max_iter=2000,

                random_state=
                RANDOM_STATE

            )

        )

    ])

)


# ============================================================
# E2 — SEMANTIC
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "E2 — SEMANTIC"
)

print(
    "=" * 80
)


# ------------------------------------------------------------
# IMPORTANT:
# Calculate unique_answer_count and majority_agreement
# directly from the five semantic answers.
# ------------------------------------------------------------

answer_columns = [

    "answer_1",

    "answer_2",

    "answer_3",

    "answer_4",

    "answer_5"

]


def normalize_answer(answer):

    if pd.isna(answer):

        return ""

    return (

        str(answer)

        .strip()

        .lower()

    )


unique_answer_counts = []

majority_agreements = []


for _, row in semantic_df.iterrows():

    answers = [

        normalize_answer(
            row[column]
        )

        for column in answer_columns

    ]


    # Remove empty answers

    answers = [

        answer

        for answer in answers

        if answer != ""

    ]


    if len(answers) == 0:

        unique_count = 0

        majority_agreement = 0.0

    else:

        counts = Counter(
            answers
        )


        unique_count = len(
            counts
        )


        largest_group = max(
            counts.values()
        )


        majority_agreement = (

            largest_group

            /

            len(answers)

        )


    unique_answer_counts.append(
        unique_count
    )

    majority_agreements.append(
        majority_agreement
    )


# ------------------------------------------------------------
# Add calculated features
# ------------------------------------------------------------

semantic_df[
    "unique_answer_count"
] = unique_answer_counts


semantic_df[
    "majority_agreement"
] = majority_agreements


# ------------------------------------------------------------
# Put semantic rows in E3 ID order
# ------------------------------------------------------------

semantic_ordered = (

    semantic_df

    .set_index("id")

    .loc[e3_df["id"]]

    .reset_index()

)


# ------------------------------------------------------------
# E2 feature set
# ------------------------------------------------------------

semantic_features = [

    "semantic_entropy",

    "semantic_consistency",

    "unique_answer_count",

    "majority_agreement"

]


X_semantic = semantic_ordered[
    semantic_features
].astype(float)


print(
    "\nE2 features:"
)

print(
    semantic_features
)


print(
    "\nFirst 5 E2 feature rows:"
)

print(
    X_semantic.head()
    .to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Evaluate E2
# ------------------------------------------------------------

evaluate_model(

    "E2 — Semantic",

    X_semantic,

    y,

    Pipeline([

        (
            "scaler",

            StandardScaler()

        ),

        (
            "classifier",

            LogisticRegression(

                max_iter=2000,

                random_state=
                RANDOM_STATE

            )

        )

    ])

)


# ============================================================
# E3 — HYBRID LOGISTIC REGRESSION
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "E3 — HYBRID LOGISTIC REGRESSION"
)

print(
    "=" * 80
)


hybrid_features = [

    "hallucination_probability",

    "semantic_consistency",

    "semantic_entropy",

    "unique_answer_count",

    "majority_agreement"

]


e3_ordered = (

    e3_df

    .set_index("id")

    .loc[e3_df["id"]]

    .reset_index()

)


X_hybrid = e3_ordered[
    hybrid_features
].astype(float)


print(
    "\nE3 features:"
)

for feature in hybrid_features:

    print(
        f"  {feature}"
    )


evaluate_model(

    "E3 — Hybrid LR",

    X_hybrid,

    y,

    Pipeline([

        (
            "scaler",

            StandardScaler()

        ),

        (
            "classifier",

            LogisticRegression(

                max_iter=1000,

                random_state=
                RANDOM_STATE

            )

        )

    ])

)


# ============================================================
# E3 — HYBRID RANDOM FOREST
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "E3 — HYBRID RANDOM FOREST"
)

print(
    "=" * 80
)


evaluate_model(

    "E3 — Hybrid RF",

    X_hybrid,

    y,

    RandomForestClassifier(

        n_estimators=200,

        max_depth=3,

        min_samples_leaf=5,

        class_weight="balanced",

        random_state=
        RANDOM_STATE

    )

)


# ============================================================
# FINAL RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


metric_columns = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1",

    "ROC-AUC",

    "PR-AUC"

]


results_df[
    metric_columns
] = (

    results_df[
        metric_columns
    ].round(4)

)


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "FINAL E1 vs E2 vs E3 RESULTS"
)

print(
    "=" * 80
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

metrics_output = os.path.join(

    OUTPUT_DIR,

    "final_metrics_500.csv"

)


results_df.to_csv(

    metrics_output,

    index=False

)


# ============================================================
# SAVE OOF PREDICTIONS
# ============================================================

oof_df = pd.DataFrame({

    "id":
        e3_df["id"].values,

    "actual_label":
        y.values

})


# ------------------------------------------------------------
# Convert model names into safe column prefixes
# ------------------------------------------------------------

safe_names = {

    "E1 — WEPR":
        "e1_wepr",

    "E2 — Semantic":
        "e2_semantic",

    "E3 — Hybrid LR":
        "e3_hybrid_lr",

    "E3 — Hybrid RF":
        "e3_hybrid_rf"

}


for name in oof_probabilities:

    prefix = safe_names[
        name
    ]


    oof_df[
        f"{prefix}_probability"
    ] = (

        oof_probabilities[
            name
        ]

    )


    oof_df[
        f"{prefix}_prediction"
    ] = (

        oof_predictions[
            name
        ]

    )


oof_output = os.path.join(

    OUTPUT_DIR,

    "oof_predictions_500.csv"

)


oof_df.to_csv(

    oof_output,

    index=False

)


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

for name, cm in (
    confusion_matrices.items()
):

    prefix = safe_names[
        name
    ]


    cm_df = pd.DataFrame(

        cm,

        index=[
            "Actual Correct",
            "Actual Hallucination"
        ],

        columns=[
            "Predicted Correct",
            "Predicted Hallucination"
        ]

    )


    cm_df.to_csv(

        os.path.join(

            OUTPUT_DIR,

            f"{prefix}_confusion_matrix.csv"

        )

    )


# ============================================================
# BEST MODEL PER METRIC
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "BEST METHOD BY METRIC"
)

print(
    "=" * 80
)


for metric in metric_columns:

    best_index = results_df[
        metric
    ].idxmax()


    best_method = results_df.loc[
        best_index,
        "Method"
    ]


    best_score = results_df.loc[
        best_index,
        metric
    ]


    print(
        f"\n{metric}:"
    )

    print(
        f"  {best_method}"
    )

    print(
        f"  Score: {best_score:.4f}"
    )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "FINAL EVALUATION COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nSaved files:"
)

print(
    f"Metrics:"
)

print(
    metrics_output
)

print(
    f"\nOOF predictions:"
)

print(
    oof_output
)

print(
    "\nConfusion matrices:"
)

print(
    OUTPUT_DIR
)