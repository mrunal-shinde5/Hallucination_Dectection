import pandas as pd
import requests
import os
import json

from semantic_consistency import calculate_semantic_consistency


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "../data/triviaqa_100.csv"

OUTPUT_FILE = (
    "../results/triviaqa_100_semantic_results.csv"
)

QWEN_URL = "http://127.0.0.1:8080/completion"

# FINAL E2 DATASET
N_QUESTIONS = 100

# Number of stochastic generations per question
N_GENERATIONS = 5

# Sampling temperature
TEMPERATURE = 0.7


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

df = df.head(N_QUESTIONS)

print(
    f"Loaded {len(df)} questions."
)

print(
    f"Generating {N_GENERATIONS} answers "
    f"per question."
)

print(
    f"Total Qwen generations required: "
    f"{len(df) * N_GENERATIONS}"
)


# ============================================================
# PROCESS QUESTIONS
# ============================================================

results = []


for index, row in df.iterrows():

    question_number = index + 1

    question = row["question"]

    reference_answer = row["reference_answer"]

    question_id = row["id"]


    print("\n" + "=" * 70)

    print(
        f"Question {question_number}/{len(df)}"
    )

    print("=" * 70)

    print(
        f"ID: {question_id}"
    )

    print(
        f"Question: {question}"
    )


    # --------------------------------------------------------
    # Generate 5 stochastic answers
    # --------------------------------------------------------

    answers = []

    raw_answers = []

    generation_errors = []


    for generation in range(N_GENERATIONS):

        print(
            f"\nGenerating answer "
            f"{generation + 1}/{N_GENERATIONS}..."
        )


        prompt = (
            "Answer the following question with only the "
            "shortest answer you believe is correct. "
            "Do not explain your answer. "
            "Do not repeat the question. "
            "Do not add unnecessary text.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )


        payload = {

            "prompt": prompt,

            "n_predict": 15,

            "temperature": TEMPERATURE,

            "n_probs": 0,

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


            raw_answer = (
                llama_result
                .get("content", "")
                .strip()
            )


            # ------------------------------------------------
            # Keep the first line as the actual answer
            # ------------------------------------------------

            generated_answer = (
                raw_answer
                .split("\n")[0]
                .strip()
            )


            if generated_answer:

                answers.append(
                    generated_answer
                )

                raw_answers.append(
                    raw_answer
                )


                print(
                    f"Answer {generation + 1}: "
                    f"{generated_answer}"
                )


            else:

                print(
                    f"Answer {generation + 1}: "
                    f"EMPTY"
                )

                generation_errors.append(
                    f"Generation {generation + 1}: "
                    f"empty response"
                )


        except Exception as e:

            print(
                f"ERROR generating answer "
                f"{generation + 1}: {e}"
            )

            generation_errors.append(
                f"Generation {generation + 1}: {str(e)}"
            )


    # --------------------------------------------------------
    # Check number of valid answers
    # --------------------------------------------------------

    if len(answers) < 2:

        print(
            "\nERROR: Not enough valid answers "
            "for semantic analysis."
        )


        results.append({

            "id": question_id,

            "question": question,

            "reference_answer":
                reference_answer,

            "answer_1":
                answers[0]
                if len(answers) > 0
                else "",

            "answer_2":
                answers[1]
                if len(answers) > 1
                else "",

            "answer_3":
                answers[2]
                if len(answers) > 2
                else "",

            "answer_4":
                answers[3]
                if len(answers) > 3
                else "",

            "answer_5":
                answers[4]
                if len(answers) > 4
                else "",

            "semantic_entropy": None,

            "normalized_entropy": None,

            "semantic_consistency": None,

            "clusters": "",

            "generation_errors":
                " | ".join(generation_errors)

        })

        continue


    # --------------------------------------------------------
    # Calculate semantic consistency
    # --------------------------------------------------------

    print(
        "\nCalculating semantic consistency..."
    )


    try:

        semantic_result = (
            calculate_semantic_consistency(
                answers,
                threshold=0.5
            )
        )


        clusters = (
            semantic_result["clusters"]
        )


        semantic_entropy = (
            semantic_result[
                "semantic_entropy"
            ]
        )


        normalized_entropy = (
            semantic_result[
                "normalized_entropy"
            ]
        )


        semantic_consistency = (
            semantic_result[
                "semantic_consistency"
            ]
        )


    except Exception as e:

        print(
            f"ERROR calculating semantic "
            f"consistency: {e}"
        )


        results.append({

            "id": question_id,

            "question": question,

            "reference_answer":
                reference_answer,

            "answer_1":
                answers[0]
                if len(answers) > 0
                else "",

            "answer_2":
                answers[1]
                if len(answers) > 1
                else "",

            "answer_3":
                answers[2]
                if len(answers) > 2
                else "",

            "answer_4":
                answers[3]
                if len(answers) > 3
                else "",

            "answer_5":
                answers[4]
                if len(answers) > 4
                else "",

            "semantic_entropy": None,

            "normalized_entropy": None,

            "semantic_consistency": None,

            "clusters": "",

            "generation_errors":
                " | ".join(generation_errors),

        })

        continue


    # --------------------------------------------------------
    # Display semantic results
    # --------------------------------------------------------

    print(
        "\nGenerated answers:"
    )


    for i, answer in enumerate(answers):

        print(
            f"{i}: {answer}"
        )


    print(
        "\nSemantic clusters:"
    )


    for cluster in clusters:

        print(
            cluster
        )


    print(
        "\nSemantic entropy:",
        semantic_entropy
    )


    print(
        "Normalized entropy:",
        normalized_entropy
    )


    print(
        "Semantic consistency:",
        semantic_consistency
    )


    # --------------------------------------------------------
    # Save question result
    # --------------------------------------------------------

    results.append({

        "id": question_id,

        "question": question,

        "reference_answer":
            reference_answer,

        "answer_1":
            answers[0]
            if len(answers) > 0
            else "",

        "answer_2":
            answers[1]
            if len(answers) > 1
            else "",

        "answer_3":
            answers[2]
            if len(answers) > 2
            else "",

        "answer_4":
            answers[3]
            if len(answers) > 3
            else "",

        "answer_5":
            answers[4]
            if len(answers) > 4
            else "",

        "semantic_entropy":
            semantic_entropy,

        "normalized_entropy":
            normalized_entropy,

        "semantic_consistency":
            semantic_consistency,

        "clusters":
            json.dumps(
                clusters
            ),

        "generation_errors":
            " | ".join(
                generation_errors
            )

    })


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    print(
        f"\nCompleted "
        f"{question_number}/{len(df)} questions."
    )


    # Save progress after every question.
    # This protects against losing all results if
    # the script stops midway.

    progress_df = pd.DataFrame(
        results
    )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    progress_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


# ============================================================
# FINAL SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "E2 — 100 QUESTION SEMANTIC EXPERIMENT COMPLETE"
)

print("=" * 70)


print(
    f"\nQuestions processed: "
    f"{len(results_df)}"
)


successful = (
    results_df[
        "semantic_consistency"
    ]
    .notna()
    .sum()
)


failed = (
    len(results_df) -
    successful
)


print(
    f"Successful semantic calculations: "
    f"{successful}"
)


print(
    f"Failed semantic calculations: "
    f"{failed}"
)


print(
    f"\nResults saved to:"
)


print(
    OUTPUT_FILE
)