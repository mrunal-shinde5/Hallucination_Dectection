import pandas as pd
import requests
import os
import re
import numpy as np
import torch

from llama_to_artefactual import convert_llama_response
from artefactual.scoring.base_detector import wepr

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "../data/triviaqa_500.csv"

WEPR_OUTPUT = (
    "../results/triviaqa_500_wepr_results.csv"
)

SEMANTIC_OUTPUT = (
    "../results/triviaqa_500_semantic_results_labelled.csv"
)

COMBINED_OUTPUT = (
    "../results/triviaqa_500_combined_features.csv"
)

QWEN_URL = "http://127.0.0.1:8080/completion"

N_QUESTIONS = 500

N_SEMANTIC_SAMPLES = 5

# Qwen sampling temperature for semantic experiment
SEMANTIC_TEMPERATURE = 1.0

# WEPR must remain deterministic
WEPR_TEMPERATURE = 0

# Number of top token probabilities requested
WEPR_TOP_LOGPROBS = 15

# NLI model
NLI_MODEL_NAME = (
    "cross-encoder/nli-deberta-v3-base"
)

NLI_THRESHOLD = 0.80


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    "../results",
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

df = df.head(
    N_QUESTIONS
)

print(
    f"Loaded {len(df)} questions."
)


# ============================================================
# LOAD WEPR DETECTOR
# ============================================================

print(
    "\nLoading WEPR detector..."
)

detector = wepr(
    "chicham/artefactual-wepr-falcon3",
    k=15
)

print(
    "WEPR detector loaded."
)


# ============================================================
# LOAD NLI MODEL
# ============================================================

print(
    "\nLoading NLI model..."
)

nli_tokenizer = (
    AutoTokenizer.from_pretrained(
        NLI_MODEL_NAME
    )
)

nli_model = (
    AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL_NAME
    )
)

nli_model.eval()

print(
    "NLI model loaded."
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).lower().strip()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# CLEAN ANSWER
# ============================================================

def clean_answer(text):

    if not text:
        return ""

    text = str(text).strip()

    text = re.sub(
        r"^(answer|response)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Only take first non-empty line.
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if lines:
        return lines[0]

    return ""


# ============================================================
# QWEN REQUEST
# ============================================================

def qwen_request(
    prompt,
    temperature=0,
    n_predict=15,
    n_probs=None
):

    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "stream": False
    }

    if n_probs is not None:

        payload["n_probs"] = n_probs

    response = requests.post(
        QWEN_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# NLI ENTAILMENT PROBABILITY
# ============================================================

def entailment_probability(
    text_a,
    text_b
):

    if not text_a or not text_b:

        return 0.0

    if (
        normalize_text(text_a)
        ==
        normalize_text(text_b)
    ):

        return 1.0

    encoded = nli_tokenizer(
        text_a,
        text_b,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():

        logits = nli_model(
            **encoded
        ).logits

    probabilities = torch.softmax(
        logits,
        dim=-1
    )[0]

    entailment_index = None

    for idx, label in (
        nli_model.config.id2label.items()
    ):

        if "entail" in label.lower():

            entailment_index = int(idx)

            break

    if entailment_index is None:

        raise RuntimeError(
            "Could not identify NLI entailment class."
        )

    return float(
        probabilities[
            entailment_index
        ]
    )


# ============================================================
# BIDIRECTIONAL SEMANTIC EQUIVALENCE
# ============================================================

def semantic_equivalence(
    text_a,
    text_b
):

    if (
        normalize_text(text_a)
        ==
        normalize_text(text_b)
    ):

        return 1

    p_ab = entailment_probability(
        text_a,
        text_b
    )

    p_ba = entailment_probability(
        text_b,
        text_a
    )

    return int(
        p_ab >= NLI_THRESHOLD
        and
        p_ba >= NLI_THRESHOLD
    )


# ============================================================
# BUILD SEMANTIC CLUSTERS
# ============================================================

def build_semantic_clusters(
    answers
):

    n = len(answers)

    assigned = [
        False
        for _ in range(n)
    ]

    clusters = []

    for i in range(n):

        if assigned[i]:
            continue

        cluster = [i]

        assigned[i] = True

        for j in range(
            i + 1,
            n
        ):

            if assigned[j]:
                continue

            equivalent = semantic_equivalence(
                answers[i],
                answers[j]
            )

            if equivalent:

                cluster.append(j)

                assigned[j] = True

        clusters.append(
            cluster
        )

    return clusters


# ============================================================
# SEMANTIC ENTROPY
# ============================================================

def calculate_semantic_entropy(
    clusters,
    total_answers
):

    if not clusters:
        return 0.0

    entropy = 0.0

    for cluster in clusters:

        probability = (
            len(cluster)
            /
            total_answers
        )

        if probability > 0:

            entropy -= (
                probability
                *
                np.log(probability)
            )

    return float(
        entropy
    )


# ============================================================
# NORMALIZED SEMANTIC ENTROPY
# ============================================================

def calculate_normalized_entropy(
    entropy,
    number_of_clusters
):

    if number_of_clusters <= 1:

        return 0.0

    max_entropy = np.log(
        number_of_clusters
    )

    if max_entropy <= 0:

        return 0.0

    normalized = (
        entropy
        /
        max_entropy
    )

    # Numerical safety.
    normalized = min(
        1.0,
        max(
            0.0,
            normalized
        )
    )

    return float(
        normalized
    )


# ============================================================
# SEMANTIC CONSISTENCY
# ============================================================

def calculate_semantic_consistency(
    normalized_entropy
):

    consistency = (
        1.0
        -
        normalized_entropy
    )

    # Numerical safety.
    consistency = max(
        0.0,
        min(
            1.0,
            consistency
        )
    )

    return float(
        consistency
    )


# ============================================================
# ACTUAL LABEL
# ============================================================

def determine_actual_label(
    generated_answer,
    reference_answer
):

    generated = normalize_text(
        generated_answer
    )

    reference = normalize_text(
        reference_answer
    )

    if not generated:

        return 1

    if not reference:

        return 1

    # Direct match
    if generated == reference:

        return 0

    # Handle multiple reference answers.
    reference_parts = re.split(
        r"\s*\|\|\s*",
        reference
    )

    for ref in reference_parts:

        if (
            generated
            ==
            normalize_text(ref)
        ):

            return 0

    return 1


# ============================================================
# WEPR RESULTS
# ============================================================

wepr_results = []


# ============================================================
# SEMANTIC RESULTS
# ============================================================

semantic_results = []


# ============================================================
# MAIN EXPERIMENT
# ============================================================

for index, row in df.iterrows():

    question_number = (
        index + 1
    )

    question = str(
        row["question"]
    )

    reference_answer = str(
        row["reference_answer"]
    )

    question_id = row["id"]


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Question "
        f"{question_number}/{len(df)}"
    )

    print(
        "=" * 70
    )

    print(
        "Question:"
    )

    print(
        question
    )

    print(
        "\nReference answer:"
    )

    print(
        reference_answer
    )


    # ========================================================
    # WEPR
    # ========================================================

    print(
        "\nRunning WEPR..."
    )


    prompt = (
        "Answer the following question with only "
        "the shortest correct answer. "
        "Do not explain your answer. "
        "Do not repeat the question. "
        "Do not add any additional text.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


    payload = {
        "prompt": prompt,
        "n_predict": 15,
        "temperature": WEPR_TEMPERATURE,
        "n_probs": WEPR_TOP_LOGPROBS,
        "stream": False
    }


    try:

        response = requests.post(
            QWEN_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        llama_result = response.json()

    except Exception as e:

        print(
            f"\nERROR generating WEPR response: {e}"
        )

        wepr_results.append({

            "id":
                question_id,

            "question":
                question,

            "reference_answer":
                reference_answer,

            "raw_response":
                "",

            "generated_answer":
                "",

            "non_hallucination_probability":
                None,

            "hallucination_probability":
                None,

            "error":
                str(e)

        })

        # We still continue with semantic experiment.
        llama_result = None


    # --------------------------------------------------------
    # WEPR calculation
    # --------------------------------------------------------

    if llama_result is not None:

        raw_answer = (
            llama_result.get(
                "content",
                ""
            ).strip()
        )

        generated_answer = (
            raw_answer
            .split("\n")[0]
            .strip()
        )

        print(
            "\nQwen WEPR answer:"
        )

        print(
            generated_answer
        )


        # ----------------------------------------------------
        # Convert llama.cpp response
        # ----------------------------------------------------

        try:

            artefactual_response = (
                convert_llama_response(
                    llama_result
                )
            )

        except Exception as e:

            print(
                f"\nERROR converting response: {e}"
            )

            wepr_results.append({

                "id":
                    question_id,

                "question":
                    question,

                "reference_answer":
                    reference_answer,

                "raw_response":
                    raw_answer,

                "generated_answer":
                    generated_answer,

                "non_hallucination_probability":
                    None,

                "hallucination_probability":
                    None,

                "error":
                    f"Conversion error: {e}"

            })

        else:

            # ------------------------------------------------
            # WEPR prediction
            # ------------------------------------------------

            try:

                score = (
                    detector.predict_proba(
                        artefactual_response
                    )
                )

                non_hallucination_probability = (
                    float(score[0][0])
                )

                hallucination_probability = (
                    float(score[0][1])
                )

                print(
                    "\nWEPR hallucination probability:"
                    f" {hallucination_probability:.4f}"
                )

                wepr_results.append({

                    "id":
                        question_id,

                    "question":
                        question,

                    "reference_answer":
                        reference_answer,

                    "raw_response":
                        raw_answer,

                    "generated_answer":
                        generated_answer,

                    "non_hallucination_probability":
                        non_hallucination_probability,

                    "hallucination_probability":
                        hallucination_probability,

                    "error":
                        ""

                })

            except Exception as e:

                print(
                    f"\nERROR running WEPR: {e}"
                )

                wepr_results.append({

                    "id":
                        question_id,

                    "question":
                        question,

                    "reference_answer":
                        reference_answer,

                    "raw_response":
                        raw_answer,

                    "generated_answer":
                        generated_answer,

                    "non_hallucination_probability":
                        None,

                    "hallucination_probability":
                        None,

                    "error":
                        f"WEPR error: {e}"

                })


    # ========================================================
    # ACTUAL LABEL
    # ========================================================

    actual_label = determine_actual_label(
        generated_answer
        if llama_result is not None
        else "",
        reference_answer
    )


    # ========================================================
    # SEMANTIC CONSISTENCY
    # ========================================================

    print(
        "\nGenerating semantic samples..."
    )

    semantic_answers = []


    for sample_number in range(
        N_SEMANTIC_SAMPLES
    ):

        print(
            f"Semantic answer "
            f"{sample_number + 1}/"
            f"{N_SEMANTIC_SAMPLES}"
        )


        semantic_prompt = (
            "Answer the following question with "
            "only the shortest correct answer. "
            "Do not explain your answer. "
            "Do not repeat the question. "
            "Do not add any additional text.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )


        try:

            semantic_result = qwen_request(
                semantic_prompt,
                temperature=SEMANTIC_TEMPERATURE,
                n_predict=30
            )

            semantic_answer = clean_answer(
                semantic_result.get(
                    "content",
                    ""
                )
            )

        except Exception as e:

            print(
                f"Semantic generation error: {e}"
            )

            semantic_answer = ""


        semantic_answers.append(
            semantic_answer
        )


        print(
            f"  Answer: {semantic_answer}"
        )


    # ========================================================
    # CLUSTER ANSWERS
    # ========================================================

    print(
        "\nCalculating semantic clusters..."
    )

    clusters = build_semantic_clusters(
        semantic_answers
    )


    print(
        "Clusters:"
    )

    print(
        clusters
    )


    # ========================================================
    # SEMANTIC ENTROPY
    # ========================================================

    semantic_entropy = (
        calculate_semantic_entropy(
            clusters,
            len(semantic_answers)
        )
    )


    normalized_entropy = (
        calculate_normalized_entropy(
            semantic_entropy,
            len(clusters)
        )
    )


    semantic_consistency = (
        calculate_semantic_consistency(
            normalized_entropy
        )
    )


    semantic_hallucination_score = (
        normalized_entropy
    )


    cluster_sizes = [
        len(cluster)
        for cluster in clusters
    ]


    if cluster_sizes:

        largest_cluster_ratio = (
            max(cluster_sizes)
            /
            len(semantic_answers)
        )

    else:

        largest_cluster_ratio = 0.0


    unique_answer_count = len(
        set(
            normalize_text(answer)
            for answer in semantic_answers
        )
    )


    print(
        "\nSemantic entropy:"
    )

    print(
        semantic_entropy
    )


    print(
        "Normalized entropy:"
    )

    print(
        normalized_entropy
    )


    print(
        "Semantic consistency:"
    )

    print(
        semantic_consistency
    )


    # ========================================================
    # SAVE SEMANTIC RESULT
    # ========================================================

    semantic_row = {

        "id":
            question_id,

        "question":
            question,

        "reference_answer":
            reference_answer,

        "actual_label":
            actual_label,

        "answer_1":
            semantic_answers[0],

        "answer_2":
            semantic_answers[1],

        "answer_3":
            semantic_answers[2],

        "answer_4":
            semantic_answers[3],

        "answer_5":
            semantic_answers[4],

        "semantic_entropy":
            semantic_entropy,

        "normalized_entropy":
            normalized_entropy,

        "semantic_consistency":
            semantic_consistency,

        "semantic_hallucination_score":
            semantic_hallucination_score,

        "number_of_clusters":
            len(clusters),

        "largest_cluster_ratio":
            largest_cluster_ratio,

        "unique_answer_count":
            unique_answer_count

    }


    semantic_results.append(
        semantic_row
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    pd.DataFrame(
        wepr_results
    ).to_csv(
        WEPR_OUTPUT,
        index=False
    )

    pd.DataFrame(
        semantic_results
    ).to_csv(
        SEMANTIC_OUTPUT,
        index=False
    )


# ============================================================
# CREATE DATAFRAMES
# ============================================================

wepr_df = pd.DataFrame(
    wepr_results
)

semantic_df = pd.DataFrame(
    semantic_results
)


# ============================================================
# LABEL WEPR DATA
# ============================================================

if (
    "generated_answer"
    in wepr_df.columns
):

    wepr_df["actual_label"] = (
        wepr_df.apply(
            lambda row:
            determine_actual_label(
                row["generated_answer"],
                row["reference_answer"]
            ),
            axis=1
        )
    )


# ============================================================
# SAVE FINAL WEPR RESULTS
# ============================================================

wepr_df.to_csv(
    WEPR_OUTPUT,
    index=False
)


# ============================================================
# SAVE FINAL SEMANTIC RESULTS
# ============================================================

semantic_df.to_csv(
    SEMANTIC_OUTPUT,
    index=False
)


# ============================================================
# CREATE COMBINED DATASET
# ============================================================

print(
    "\nCreating combined feature dataset..."
)


combined = pd.merge(

    wepr_df[
        [
            "id",
            "question",
            "reference_answer",
            "generated_answer",
            "non_hallucination_probability",
            "hallucination_probability",
            "actual_label"
        ]
    ],

    semantic_df[
        [
            "id",
            "semantic_entropy",
            "normalized_entropy",
            "semantic_consistency",
            "semantic_hallucination_score",
            "number_of_clusters",
            "largest_cluster_ratio",
            "unique_answer_count"
        ]
    ],

    on="id",

    how="inner"
)


# ============================================================
# CHECK MERGE
# ============================================================

print(
    "\nCombined dataset:"
)

print(
    f"Rows: {len(combined)}"
)

print(
    f"Expected: {len(df)}"
)


if len(combined) != len(df):

    print(
        "\nWARNING:"
    )

    print(
        "Combined dataset does not contain "
        "all questions."
    )


# ============================================================
# SAVE COMBINED DATASET
# ============================================================

combined.to_csv(
    COMBINED_OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "100-QUESTION EXPERIMENT COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nWEPR results saved to:"
)

print(
    WEPR_OUTPUT
)


print(
    "\nSemantic results saved to:"
)

print(
    SEMANTIC_OUTPUT
)


print(
    "\nCombined features saved to:"
)

print(
    COMBINED_OUTPUT
)


print(
    "\nQuestions processed:"
)

print(
    len(df)
)


print(
    "\nWEPR successful:"
)

print(
    wepr_df[
        "hallucination_probability"
    ].notna().sum()
)


print(
    "\nWEPR failed:"
)

print(
    wepr_df[
        "hallucination_probability"
    ].isna().sum()
)


print(
    "\nLabel distribution:"
)

print(
    combined[
        "actual_label"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\n" + "=" * 70
)

print(
    "READY FOR E3 FEATURE FUSION"
)

print(
    "=" * 70
)