import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from sklearn.base import clone


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "../results/triviaqa_100_combined_features.csv"
)

OUTPUT_FILE = (
    "../results/e4_feature_ablation_results.csv"
)

N_SPLITS = 5

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

print("=" * 70)
print("E4 — FEATURE ABLATION")
print("=" * 70)

print(
    f"\nLoaded {len(df)} questions."
)


# ============================================================
# TARGET
# ============================================================

TARGET = "actual_label"

y = df[TARGET].astype(int)


# ============================================================
# FEATURE SETS
# ============================================================

feature_sets = {

    # --------------------------------------------------------
    # E1
    # --------------------------------------------------------

    "WEPR only": [

        "hallucination_probability"

    ],


    # --------------------------------------------------------
    # E2 core
    # --------------------------------------------------------

    "Semantic entropy only": [

        "semantic_entropy"

    ],


    # --------------------------------------------------------
    # E2 extended
    # --------------------------------------------------------

    "Semantic extended": [

        "semantic_entropy",

        "number_of_clusters",

        "largest_cluster_ratio",

        "unique_answer_count"

    ],


    # --------------------------------------------------------
    # WEPR + semantic core
    # --------------------------------------------------------

    "WEPR + Semantic entropy": [

        "hallucination_probability",

        "semantic_entropy"

    ],


    # --------------------------------------------------------
    # WEPR + semantic extended
    # --------------------------------------------------------

    "WEPR + Semantic extended": [

        "hallucination_probability",

        "semantic_entropy",

        "number_of_clusters",

        "largest_cluster_ratio",

        "unique_answer_count"

    ]

}


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(

    n_splits=N_SPLITS,

    shuffle=True,

    random_state=RANDOM_STATE

)


# ============================================================
# MODEL
# ============================================================

base_model = Pipeline([

    (
        "scaler",

        StandardScaler()
    ),

    (
        "classifier",

        LogisticRegression(

            max_iter=2000,

            class_weight="balanced",

            random_state=RANDOM_STATE

        )

    )

])


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# RUN ABLATION
# ============================================================

for feature_name, features in feature_sets.items():

    print(
        "\n"
        + "=" * 70
    )

    print(
        feature_name
    )

    print(
        "=" * 70
    )

    X = df[
        features
    ].copy()


    fold_metrics = []


    for fold, (
        train_idx,
        test_idx
    ) in enumerate(

        cv.split(X, y),

        start=1

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

        y_test = y.iloc[
            test_idx
        ]


        model = clone(
            base_model
        )


        model.fit(
            X_train,
            y_train
        )


        predictions = model.predict(
            X_test
        )


        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )


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


        pr_auc = (
            average_precision_score(
                y_test,
                probabilities
            )
        )


        fold_metrics.append({

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "roc_auc": roc_auc,

            "pr_auc": pr_auc

        })


        print(

            f"Fold {fold}: "

            f"ROC-AUC={roc_auc:.4f}, "

            f"PR-AUC={pr_auc:.4f}, "

            f"F1={f1:.4f}"

        )


    # ========================================================
    # MEAN / STD
    # ========================================================

    fold_df = pd.DataFrame(
        fold_metrics
    )


    results.append({

        "feature_set":
            feature_name,

        "features":
            ", ".join(features),

        "accuracy_mean":
            fold_df[
                "accuracy"
            ].mean(),

        "accuracy_std":
            fold_df[
                "accuracy"
            ].std(),

        "precision_mean":
            fold_df[
                "precision"
            ].mean(),

        "precision_std":
            fold_df[
                "precision"
            ].std(),

        "recall_mean":
            fold_df[
                "recall"
            ].mean(),

        "recall_std":
            fold_df[
                "recall"
            ].std(),

        "f1_mean":
            fold_df[
                "f1"
            ].mean(),

        "f1_std":
            fold_df[
                "f1"
            ].std(),

        "roc_auc_mean":
            fold_df[
                "roc_auc"
            ].mean(),

        "roc_auc_std":
            fold_df[
                "roc_auc"
            ].std(),

        "pr_auc_mean":
            fold_df[
                "pr_auc"
            ].mean(),

        "pr_auc_std":
            fold_df[
                "pr_auc"
            ].std()

    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "E4 FEATURE ABLATION RESULTS"
)

print(
    "=" * 70
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# BEST RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "BEST RESULTS"
)

print(
    "=" * 70
)


for metric in [

    "accuracy_mean",

    "f1_mean",

    "roc_auc_mean",

    "pr_auc_mean"

]:

    best_idx = results_df[
        metric
    ].idxmax()


    best = results_df.loc[
        best_idx
    ]


    print(
        f"\nBest {metric}:"
    )

    print(
        f"Feature set: "
        f"{best['feature_set']}"
    )

    print(
        f"Score: "
        f"{best[metric]:.4f}"
    )


# ============================================================
# SPECIFIC WEPR CONTRIBUTION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "WEPR CONTRIBUTION ANALYSIS"
)

print(
    "=" * 70
)


semantic_only = results_df[
    results_df[
        "feature_set"
    ]
    ==
    "Semantic extended"
].iloc[0]


combined = results_df[
    results_df[
        "feature_set"
    ]
    ==
    "WEPR + Semantic extended"
].iloc[0]


print(
    "\nSemantic extended ROC-AUC:"
)

print(
    f"{semantic_only['roc_auc_mean']:.4f}"
)


print(
    "WEPR + Semantic extended ROC-AUC:"
)

print(
    f"{combined['roc_auc_mean']:.4f}"
)


roc_difference = (
    combined["roc_auc_mean"]
    -
    semantic_only["roc_auc_mean"]
)


print(
    "\nWEPR contribution to ROC-AUC:"
)

print(
    f"{roc_difference:+.4f}"
)


print(
    "\nSemantic extended PR-AUC:"
)

print(
    f"{semantic_only['pr_auc_mean']:.4f}"
)


print(
    "WEPR + Semantic extended PR-AUC:"
)

print(
    f"{combined['pr_auc_mean']:.4f}"
)


pr_difference = (
    combined["pr_auc_mean"]
    -
    semantic_only["pr_auc_mean"]
)


print(
    "\nWEPR contribution to PR-AUC:"
)

print(
    f"{pr_difference:+.4f}"
)


print(
    "\n"
    + "=" * 70
)

print(
    "E4 COMPLETE"
)

print(
    "=" * 70
)

print(
    f"\nResults saved to:"
)

print(
    OUTPUT_FILE
)