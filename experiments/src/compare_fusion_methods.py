import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "../results/triviaqa_100_E1_E2_combined.csv"

OUTPUT_FILE = "../results/fusion_comparison_results.csv"

RANDOM_STATE = 42

N_SPLITS = 5


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(
    f"Loaded {len(df)} questions."
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "actual_label",
    "hallucination_probability",
    "semantic_hallucination_score"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

df = df.dropna(
    subset=required_columns
).copy()


# ============================================================
# FEATURES
# ============================================================

wepr = df[
    "hallucination_probability"
].astype(float).values


semantic = df[
    "semantic_hallucination_score"
].astype(float).values


y = df[
    "actual_label"
].astype(int).values


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities >= threshold
    ).astype(int)


    return {

        "Accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "ROC-AUC":
            roc_auc_score(
                y_true,
                probabilities
            ),

        "PR-AUC":
            average_precision_score(
                y_true,
                probabilities
            ),

        "Precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )

    }


# ============================================================
# STORE RESULTS
# ============================================================

results = []


# ============================================================
# E1 — WEPR BASELINE
# ============================================================

wepr_metrics = calculate_metrics(
    y,
    wepr
)


results.append({

    "Method": "WEPR",

    **wepr_metrics

})


# ============================================================
# E2 — SEMANTIC BASELINE
# ============================================================

semantic_metrics = calculate_metrics(
    y,
    semantic
)


results.append({

    "Method": "Semantic",

    **semantic_metrics

})


# ============================================================
# E3a — WEIGHTED FUSION
# ============================================================
#
# We test:
#
# 100% WEPR / 0% Semantic
# 90% WEPR  / 10% Semantic
# ...
# 50% WEPR  / 50% Semantic
# ...
# 0% WEPR   / 100% Semantic
#
# The weight is selected INSIDE each training fold
# using the training data only.
#
# This prevents test-set leakage.
# ============================================================

weighted_oof = np.zeros(len(df))


for train_idx, test_idx in cv.split(
    np.zeros(len(y)),
    y
):

    train_y = y[train_idx]

    train_wepr = wepr[train_idx]

    train_semantic = semantic[train_idx]

    test_wepr = wepr[test_idx]

    test_semantic = semantic[test_idx]


    best_weight = None

    best_train_auc = -np.inf


    # Test WEPR weights from 0.0 to 1.0.
    for weight in np.linspace(
        0.0,
        1.0,
        101
    ):

        train_score = (
            weight * train_wepr
            +
            (1.0 - weight)
            * train_semantic
        )


        auc = roc_auc_score(
            train_y,
            train_score
        )


        if auc > best_train_auc:

            best_train_auc = auc

            best_weight = weight


    # Apply the selected weight ONLY
    # to the held-out fold.

    weighted_oof[test_idx] = (
        best_weight * test_wepr
        +
        (1.0 - best_weight)
        * test_semantic
    )


weighted_metrics = calculate_metrics(
    y,
    weighted_oof
)


results.append({

    "Method": "Weighted Fusion",

    **weighted_metrics

})


# ============================================================
# E3b — LOGISTIC REGRESSION
# ============================================================

X = np.column_stack(
    [
        wepr,
        semantic
    ]
)


logistic_model = Pipeline(
    steps=[

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )

    ]
)


logistic_oof = np.zeros(
    len(y)
)


for train_idx, test_idx in cv.split(
    X,
    y
):

    X_train = X[
        train_idx
    ]

    X_test = X[
        test_idx
    ]

    y_train = y[
        train_idx
    ]


    logistic_model.fit(
        X_train,
        y_train
    )


    logistic_oof[test_idx] = (
        logistic_model
        .predict_proba(X_test)[:, 1]
    )


logistic_metrics = calculate_metrics(
    y,
    logistic_oof
)


results.append({

    "Method":
        "Logistic Regression",

    **logistic_metrics

})


# ============================================================
# E3c — RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(

    n_estimators=200,

    max_depth=3,

    min_samples_leaf=5,

    random_state=RANDOM_STATE,

    class_weight="balanced"

)


rf_oof = np.zeros(
    len(y)
)


for train_idx, test_idx in cv.split(
    X,
    y
):

    X_train = X[
        train_idx
    ]

    X_test = X[
        test_idx
    ]

    y_train = y[
        train_idx
    ]


    rf_model.fit(
        X_train,
        y_train
    )


    rf_oof[test_idx] = (
        rf_model
        .predict_proba(X_test)[:, 1]
    )


rf_metrics = calculate_metrics(
    y,
    rf_oof
)


results.append({

    "Method":
        "Random Forest",

    **rf_metrics

})


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


# Round for readability.

results_df[
    [
        "Accuracy",
        "ROC-AUC",
        "PR-AUC",
        "Precision",
        "Recall",
        "F1"
    ]
] = results_df[
    [
        "Accuracy",
        "ROC-AUC",
        "PR-AUC",
        "Precision",
        "Recall",
        "F1"
    ]
].round(4)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 80)

print(
    "E4 — FUSION METHOD COMPARISON"
)

print("=" * 80)

print()

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# FIND BEST METHODS
# ============================================================

print("\n" + "=" * 80)

print(
    "BEST METHOD BY METRIC"
)

print("=" * 80)


for metric in [
    "Accuracy",
    "ROC-AUC",
    "PR-AUC",
    "Precision",
    "Recall",
    "F1"
]:

    best_idx = (
        results_df[metric]
        .idxmax()
    )

    best_method = (
        results_df.loc[
            best_idx,
            "Method"
        ]
    )

    best_value = (
        results_df.loc[
            best_idx,
            metric
        ]
    )


    print(
        f"{metric:<12}: "
        f"{best_method:<20} "
        f"{best_value:.4f}"
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nSaved comparison to:"
)

print(
    OUTPUT_FILE
)