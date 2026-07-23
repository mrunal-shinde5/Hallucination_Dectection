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

OUTPUT_FILE = (
    "../results/triviaqa_100_multifeature_results.csv"
)

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
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "id",
    "actual_label",
    "hallucination_probability",
    "semantic_consistency",
    "semantic_entropy",
    "answer_1",
    "answer_2",
    "answer_3",
    "answer_4",
    "answer_5"
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
# NORMALIZE ANSWERS
# ============================================================

def normalize_answer(answer):

    if pd.isna(answer):

        return ""

    return (
        str(answer)
        .strip()
        .lower()
    )


answer_columns = [
    "answer_1",
    "answer_2",
    "answer_3",
    "answer_4",
    "answer_5"
]


# ============================================================
# CALCULATE SURFACE-LEVEL FEATURES
# ============================================================

unique_counts = []

majority_agreements = []


for _, row in df.iterrows():

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
            largest_group /
            len(answers)
        )


    unique_counts.append(
        unique_count
    )

    majority_agreements.append(
        majority_agreement
    )


df["unique_answer_count"] = (
    unique_counts
)


df["majority_agreement"] = (
    majority_agreements
)


# ============================================================
# DISPLAY NEW FEATURES
# ============================================================

print(
    "\nNew surface-level features:"
)

print(
    df[
        [
            "id",
            "unique_answer_count",
            "majority_agreement"
        ]
    ].head(10).to_string(
        index=False
    )
)


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

feature_columns = [

    "hallucination_probability",

    "semantic_consistency",

    "semantic_entropy",

    "unique_answer_count",

    "majority_agreement"

]


df = df.dropna(
    subset=feature_columns + [
        "actual_label"
    ]
).copy()


# ============================================================
# CREATE FEATURE MATRIX
# ============================================================

X = df[
    feature_columns
].astype(float)


y = df[
    "actual_label"
].astype(int)


print(
    f"\nQuestions available: "
    f"{len(df)}"
)


print(
    "\nFeatures used:"
)

for feature in feature_columns:

    print(
        f"  - {feature}"
    )


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
# RESULTS
# ============================================================

results = []


# ============================================================
# MULTI-FEATURE LOGISTIC REGRESSION
# ============================================================

print(
    "\nRunning multi-feature Logistic Regression..."
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

    X_train = X.iloc[
        train_idx
    ]

    X_test = X.iloc[
        test_idx
    ]

    y_train = y.iloc[
        train_idx
    ]


    logistic_model.fit(
        X_train,
        y_train
    )


    logistic_oof[test_idx] = (
        logistic_model
        .predict_proba(
            X_test
        )[:, 1]
    )


logistic_metrics = calculate_metrics(
    y,
    logistic_oof
)


results.append({

    "Method":
        "5-Feature Logistic Regression",

    **logistic_metrics

})


# ============================================================
# MULTI-FEATURE RANDOM FOREST
# ============================================================

print(
    "\nRunning multi-feature Random Forest..."
)


rf_model = RandomForestClassifier(

    n_estimators=200,

    max_depth=3,

    min_samples_leaf=5,

    class_weight="balanced",

    random_state=RANDOM_STATE

)


rf_oof = np.zeros(
    len(y)
)


for train_idx, test_idx in cv.split(
    X,
    y
):

    X_train = X.iloc[
        train_idx
    ]

    X_test = X.iloc[
        test_idx
    ]

    y_train = y.iloc[
        train_idx
    ]


    rf_model.fit(
        X_train,
        y_train
    )


    rf_oof[test_idx] = (
        rf_model
        .predict_proba(
            X_test
        )[:, 1]
    )


rf_metrics = calculate_metrics(
    y,
    rf_oof
)


results.append({

    "Method":
        "5-Feature Random Forest",

    **rf_metrics

})


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


metric_columns = [

    "Accuracy",
    "ROC-AUC",
    "PR-AUC",
    "Precision",
    "Recall",
    "F1"

]


results_df[
    metric_columns
] = results_df[
    metric_columns
].round(4)


print(
    "\n" + "=" * 80
)

print(
    "MULTI-FEATURE HYBRID RESULTS"
)

print(
    "=" * 80
)

print()

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# FIT FINAL MODELS
# ============================================================
#
# We fit on all 100 questions ONLY after the
# cross-validation evaluation has been completed.
#
# These models are useful for examining feature
# importance/coefficients.
# ============================================================


# ------------------------------------------------------------
# Logistic Regression coefficients
# ------------------------------------------------------------

final_logistic = Pipeline(

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


final_logistic.fit(
    X,
    y
)


classifier = (
    final_logistic
    .named_steps[
        "classifier"
    ]
)


coefficients = (
    classifier
    .coef_[0]
)


coefficient_df = pd.DataFrame({

    "Feature":
        feature_columns,

    "Coefficient":
        coefficients

})


coefficient_df[
    "Absolute_Coefficient"
] = (
    coefficient_df[
        "Coefficient"
    ].abs()
)


coefficient_df = (
    coefficient_df
    .sort_values(
        "Absolute_Coefficient",
        ascending=False
    )
)


print(
    "\n" + "=" * 80
)

print(
    "LOGISTIC REGRESSION FEATURE COEFFICIENTS"
)

print(
    "=" * 80
)

print(
    coefficient_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Random Forest feature importance
# ------------------------------------------------------------

final_rf = RandomForestClassifier(

    n_estimators=200,

    max_depth=3,

    min_samples_leaf=5,

    class_weight="balanced",

    random_state=RANDOM_STATE

)


final_rf.fit(
    X,
    y
)


importance_df = pd.DataFrame({

    "Feature":
        feature_columns,

    "Importance":
        final_rf.feature_importances_

})


importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
)


print(
    "\n" + "=" * 80
)

print(
    "RANDOM FOREST FEATURE IMPORTANCE"
)

print(
    "=" * 80
)

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# Save feature analyses separately.

coefficient_df.to_csv(
    "../results/"
    "multifeature_logistic_coefficients.csv",
    index=False
)


importance_df.to_csv(
    "../results/"
    "multifeature_rf_importance.csv",
    index=False
)


print(
    "\n" + "=" * 80
)

print(
    "FILES SAVED"
)

print(
    "=" * 80
)

print(
    "\nMain results:"
)

print(
    OUTPUT_FILE
)

print(
    "\nLogistic coefficients:"
)

print(
    "../results/"
    "multifeature_logistic_coefficients.csv"
)

print(
    "\nRandom Forest importance:"
)

print(
    "../results/"
    "multifeature_rf_importance.csv"
)