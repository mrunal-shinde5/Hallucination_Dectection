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

WEPR_FILE = "../results/triviaqa_500_wepr_results.csv"

SEMANTIC_FILE = (
    "../results/triviaqa_500_semantic_results_labelled.csv"
)

OUTPUT_FILE = (
    "../results/triviaqa_500_multifeature_results.csv"
)

LOGISTIC_COEFFICIENT_FILE = (
    "../results/"
    "triviaqa_500_multifeature_logistic_coefficients.csv"
)

RF_IMPORTANCE_FILE = (
    "../results/"
    "triviaqa_500_multifeature_rf_importance.csv"
)

RANDOM_STATE = 42

N_SPLITS = 5


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 70)
print("E3 — 5-FEATURE MULTI-FEATURE HYBRID")
print("=" * 70)

print("\nLoading WEPR results...")

wepr_df = pd.read_csv(
    WEPR_FILE
)

print(
    f"WEPR rows: {len(wepr_df)}"
)


print("\nLoading semantic results...")

semantic_df = pd.read_csv(
    SEMANTIC_FILE
)

print(
    f"Semantic rows: {len(semantic_df)}"
)


# ============================================================
# CHECK DATASET SIZES
# ============================================================

if len(wepr_df) != 500:
    raise ValueError(
        f"Expected 500 WEPR rows, "
        f"found {len(wepr_df)}"
    )

if len(semantic_df) != 500:
    raise ValueError(
        f"Expected 500 semantic rows, "
        f"found {len(semantic_df)}"
    )


# ============================================================
# MERGE WEPR + SEMANTIC
# ============================================================

print(
    "\nMerging WEPR and semantic results..."
)

df = pd.merge(
    wepr_df,
    semantic_df[
        [
            "id",
            "answer_1",
            "answer_2",
            "answer_3",
            "answer_4",
            "answer_5",
            "semantic_entropy",
            "semantic_consistency",
            "actual_label"
        ]
    ],
    on="id",
    how="inner",
    suffixes=("_wepr", "_semantic")
)


print(
    f"Merged rows: {len(df)}"
)


if len(df) != 500:

    raise ValueError(
        "Merge did not produce exactly "
        "500 rows."
    )


# ============================================================
# CHECK LABEL CONSISTENCY
# ============================================================

label_mismatch = (

    df["actual_label_wepr"]

    !=

    df["actual_label_semantic"]

).sum()


print(
    f"Label mismatches: "
    f"{label_mismatch}"
)


if label_mismatch > 0:

    raise ValueError(
        "WEPR and semantic files contain "
        "different actual labels."
    )


df["actual_label"] = (
    df["actual_label_wepr"]
    .astype(int)
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
# CALCULATE UNIQUE ANSWERS
# AND MAJORITY AGREEMENT
# ============================================================

print(
    "\nCalculating surface-level features..."
)


unique_answer_counts = []

majority_agreements = []


for _, row in df.iterrows():

    answers = [

        normalize_answer(
            row[column]
        )

        for column in answer_columns

    ]


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


# ============================================================
# ADD FEATURES
# ============================================================

df["unique_answer_count"] = (
    unique_answer_counts
)

df["majority_agreement"] = (
    majority_agreements
)


# ============================================================
# FIVE FINAL FEATURES
# ============================================================

FEATURES = [

    "hallucination_probability",

    "semantic_consistency",

    "semantic_entropy",

    "unique_answer_count",

    "majority_agreement"

]


TARGET = "actual_label"


# ============================================================
# CHECK FEATURES
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FINAL E3 FEATURES"
)

print(
    "=" * 70
)

for feature in FEATURES:

    print(
        f"  {feature}"
    )


# ============================================================
# CREATE X AND Y
# ============================================================

df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()


X = df[
    FEATURES
].astype(float)


y = df[
    TARGET
].astype(int)


print(
    f"\nFinal rows used: {len(df)}"
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print(
    "\nLabel distribution:"
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
# LOGISTIC REGRESSION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "RUNNING 5-FEATURE LOGISTIC REGRESSION"
)

print(
    "=" * 70
)


logistic_oof = np.zeros(
    len(y)
)


for fold, (
    train_idx,
    test_idx
) in enumerate(

    cv.split(X, y),

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


    model = Pipeline([

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


    model.fit(
        X_train,
        y_train
    )


    logistic_oof[
        test_idx
    ] = (

        model.predict_proba(
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
# RANDOM FOREST
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "RUNNING 5-FEATURE RANDOM FOREST"
)

print(
    "=" * 70
)


rf_oof = np.zeros(
    len(y)
)


for fold, (
    train_idx,
    test_idx
) in enumerate(

    cv.split(X, y),

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


    model = RandomForestClassifier(

        n_estimators=200,

        max_depth=3,

        min_samples_leaf=5,

        class_weight="balanced",

        random_state=
        RANDOM_STATE

    )


    model.fit(
        X_train,
        y_train
    )


    rf_oof[
        test_idx
    ] = (

        model.predict_proba(
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
# FINAL RESULTS
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
] = (

    results_df[
        metric_columns
    ].round(4)

)


print(
    "\n"
    + "=" * 80
)

print(
    "E3 — 500 QUESTION RESULTS"
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
# SAVE RESULTS
# ============================================================

results_df.to_csv(

    OUTPUT_FILE,

    index=False

)


# ============================================================
# FINAL LOGISTIC MODEL
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "LOGISTIC REGRESSION FEATURE COEFFICIENTS"
)

print(
    "=" * 70
)


final_logistic = Pipeline([

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


final_logistic.fit(
    X,
    y
)


coefficients = (

    final_logistic

    .named_steps[
        "classifier"
    ]

    .coef_[0]

)


coefficient_df = pd.DataFrame({

    "Feature":
        FEATURES,

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
    coefficient_df.to_string(
        index=False
    )
)


# ============================================================
# FINAL RANDOM FOREST
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "RANDOM FOREST FEATURE IMPORTANCE"
)

print(
    "=" * 70
)


final_rf = RandomForestClassifier(

    n_estimators=200,

    max_depth=3,

    min_samples_leaf=5,

    class_weight="balanced",

    random_state=
    RANDOM_STATE

)


final_rf.fit(
    X,
    y
)


importance_df = pd.DataFrame({

    "Feature":
        FEATURES,

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
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE FEATURE ANALYSIS
# ============================================================

coefficient_df.to_csv(

    LOGISTIC_COEFFICIENT_FILE,

    index=False

)


importance_df.to_csv(

    RF_IMPORTANCE_FILE,

    index=False

)


# ============================================================
# SAVE E3 FEATURE DATASET
# ============================================================

E3_DATASET = (
    "../results/"
    "triviaqa_500_E3_features.csv"
)


df[
    [
        "id",
        "actual_label"
    ]
    +
    FEATURES
].to_csv(

    E3_DATASET,

    index=False

)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "E3 COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nResults:"
)

print(
    OUTPUT_FILE
)

print(
    "\nE3 feature dataset:"
)

print(
    E3_DATASET
)

print(
    "\nLogistic coefficients:"
)

print(
    LOGISTIC_COEFFICIENT_FILE
)

print(
    "\nRandom Forest importance:"
)

print(
    RF_IMPORTANCE_FILE
)