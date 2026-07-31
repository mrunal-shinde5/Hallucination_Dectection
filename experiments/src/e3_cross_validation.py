import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "../results/triviaqa_100_combined_features.csv"
)

OUTPUT_FILE = (
    "../results/e3_cross_validation_results.csv"
)

N_SPLITS = 5

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("E3 — WEPR + SEMANTIC FEATURE FUSION")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nLoaded {len(df)} questions."
)


# ============================================================
# CHECK LABELS
# ============================================================

print("\nLabel distribution:")

print(
    df["actual_label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

# ------------------------------------------------------------
# E1 — WEPR
# ------------------------------------------------------------

WEPR_FEATURES = [
    "hallucination_probability"
]


# ------------------------------------------------------------
# E2 — SEMANTIC
# ------------------------------------------------------------

SEMANTIC_FEATURES = [
    "semantic_entropy",
    "normalized_entropy",
    "semantic_consistency",
    "semantic_hallucination_score",
    "number_of_clusters",
    "largest_cluster_ratio",
    "unique_answer_count"
]


# ------------------------------------------------------------
# E3 — COMBINED
# ------------------------------------------------------------

COMBINED_FEATURES = (
    WEPR_FEATURES
    +
    SEMANTIC_FEATURES
)


TARGET = "actual_label"


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    COMBINED_FEATURES
    +
    [TARGET]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

before = len(df)

df = df.dropna(
    subset=required_columns
).copy()

after = len(df)

if before != after:

    print(
        f"\nRemoved {before - after} rows "
        "containing missing feature values."
    )


# ============================================================
# TARGET
# ============================================================

y = df[TARGET].astype(int)


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

    "SVM":
        SVC(
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
}


# ============================================================
# FEATURE GROUPS
# ============================================================

feature_groups = {

    "E1 — WEPR": WEPR_FEATURES,

    "E2 — Semantic": SEMANTIC_FEATURES,

    "E3 — WEPR + Semantic": COMBINED_FEATURES

}


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    feature_name,
    model_name
):

    fold_results = []

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"{feature_name} | {model_name}"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Cross validation
    # --------------------------------------------------------

    for fold, (
        train_index,
        test_index
    ) in enumerate(
        cv.split(X, y),
        start=1
    ):

        X_train = X.iloc[
            train_index
        ]

        X_test = X.iloc[
            test_index
        ]

        y_train = y.iloc[
            train_index
        ]

        y_test = y.iloc[
            test_index
        ]


        # ----------------------------------------------------
        # Clone model
        # ----------------------------------------------------

        from sklearn.base import clone

        fold_model = clone(
            model
        )


        # ----------------------------------------------------
        # Fit
        # ----------------------------------------------------

        fold_model.fit(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predictions = (
            fold_model.predict(
                X_test
            )
        )

        probabilities = (
            fold_model.predict_proba(
                X_test
            )[:, 1]
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        pr_auc = average_precision_score(
            y_test,
            probabilities
        )


        fold_results.append({

            "feature_group":
                feature_name,

            "model":
                model_name,

            "fold":
                fold,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "roc_auc":
                roc_auc,

            "pr_auc":
                pr_auc

        })


        print(
            f"Fold {fold}: "
            f"ROC-AUC={roc_auc:.4f}, "
            f"PR-AUC={pr_auc:.4f}, "
            f"F1={f1:.4f}"
        )


    return fold_results


# ============================================================
# RUN EXPERIMENT
# ============================================================

all_results = []


for feature_name, feature_list in (
    feature_groups.items()
):

    print(
        "\n\n"
        + "#" * 70
    )

    print(
        feature_name
    )

    print(
        "#" * 70
    )


    X = df[
        feature_list
    ].copy()


    # --------------------------------------------------------
    # Standardization
    #
    # Important for:
    # - Logistic Regression
    # - SVM
    #
    # Random Forest doesn't require it, but we keep
    # preprocessing simple and consistent.
    # --------------------------------------------------------

    for model_name, model in (
        models.items()
    ):

        if model_name in [
            "Logistic Regression",
            "SVM"
        ]:

            pipeline = Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    model
                )
            ])

        else:

            pipeline = model


        fold_results = evaluate_model(
            pipeline,
            X,
            y,
            feature_name,
            model_name
        )

        all_results.extend(
            fold_results
        )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    results_df
    .groupby(
        [
            "feature_group",
            "model"
        ]
    )
    [
        [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc"
        ]
    ]
    .agg(
        [
            "mean",
            "std"
        ]
    )
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n\n"
    + "=" * 70
)

print(
    "E3 CROSS-VALIDATION SUMMARY"
)

print(
    "=" * 70
)

print(
    summary.to_string()
)


# ============================================================
# CLEAN COMPARISON TABLE
# ============================================================

comparison = (
    results_df
    .groupby(
        [
            "feature_group",
            "model"
        ]
    )
    .agg(
        accuracy_mean=(
            "accuracy",
            "mean"
        ),

        accuracy_std=(
            "accuracy",
            "std"
        ),

        precision_mean=(
            "precision",
            "mean"
        ),

        precision_std=(
            "precision",
            "std"
        ),

        recall_mean=(
            "recall",
            "mean"
        ),

        recall_std=(
            "recall",
            "std"
        ),

        f1_mean=(
            "f1",
            "mean"
        ),

        f1_std=(
            "f1",
            "std"
        ),

        roc_auc_mean=(
            "roc_auc",
            "mean"
        ),

        roc_auc_std=(
            "roc_auc",
            "std"
        ),

        pr_auc_mean=(
            "pr_auc",
            "mean"
        ),

        pr_auc_std=(
            "pr_auc",
            "std"
        )
    )
    .reset_index()
)


# ============================================================
# PRINT FINAL COMPARISON
# ============================================================

print(
    "\n\n"
    + "=" * 70
)

print(
    "FINAL E1 vs E2 vs E3 COMPARISON"
)

print(
    "=" * 70
)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# SAVE SUMMARY
# ============================================================

SUMMARY_OUTPUT = (
    "../results/"
    "e3_cross_validation_summary.csv"
)

comparison.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


print(
    "\n\nFold-level results saved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nSummary saved to:"
)

print(
    SUMMARY_OUTPUT
)


# ============================================================
# BEST RESULTS
# ============================================================

print(
    "\n\n"
    + "=" * 70
)

print(
    "BEST RESULTS"
)

print(
    "=" * 70
)


for metric in [
    "roc_auc_mean",
    "pr_auc_mean",
    "f1_mean"
]:

    best = comparison.loc[
        comparison[metric].idxmax()
    ]

    print(
        f"\nBest {metric}:"
    )

    print(
        f"Feature group: "
        f"{best['feature_group']}"
    )

    print(
        f"Model: "
        f"{best['model']}"
    )

    print(
        f"Score: "
        f"{best[metric]:.4f}"
    )


print(
    "\n"
    + "=" * 70
)

print(
    "E3 COMPLETE"
)

print(
    "=" * 70
)