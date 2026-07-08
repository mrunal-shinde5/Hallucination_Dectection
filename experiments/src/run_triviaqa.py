import pandas as pd
import requests
import os

from llama_to_artefactual import convert_llama_response
from artefactual.scoring.base_detector import wepr


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "../data/triviaqa_100.csv"
OUTPUT_FILE = "../results/triviaqa_100_results.csv"

QWEN_URL = "http://127.0.0.1:8080/completion"

N_QUESTIONS = 100


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

df = df.head(N_QUESTIONS)

print(f"Loaded {len(df)} questions.")


# ============================================================
# LOAD WEPR DETECTOR
# ============================================================

print("\nLoading WEPR detector...")

detector = wepr(
    "chicham/artefactual-wepr-falcon3",
    k=15
)

print("WEPR detector loaded.")


# ============================================================
# PROCESS QUESTIONS
# ============================================================

results = []


for index, row in df.iterrows():

    question_number = index + 1

    question = row["question"]
    reference_answer = row["reference_answer"]

    print("\n" + "=" * 70)
    print(f"Question {question_number}/{len(df)}")
    print("=" * 70)

    print("Question:")
    print(question)


    # --------------------------------------------------------
    # Create Qwen prompt
    # --------------------------------------------------------

    prompt = (
        "Answer the following question with only the shortest "
        "correct answer. "
        "Do not explain your answer. "
        "Do not repeat the question. "
        "Do not add any additional text.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


    # --------------------------------------------------------
    # Qwen request
    # --------------------------------------------------------

    payload = {
        "prompt": prompt,
        "n_predict": 15,
        "temperature": 0,
        "n_probs": 15,
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

        print(f"\nERROR generating response: {e}")

        results.append({
            "id": row["id"],
            "question": question,
            "reference_answer": reference_answer,
            "raw_response": "",
            "generated_answer": "",
            "non_hallucination_probability": None,
            "hallucination_probability": None,
            "error": str(e)
        })

        continue


    # --------------------------------------------------------
    # Store raw response
    # --------------------------------------------------------

    raw_answer = llama_result.get(
        "content",
        ""
    ).strip()


    # --------------------------------------------------------
    # Clean answer
    # --------------------------------------------------------

    generated_answer = raw_answer.split("\n")[0].strip()


    print("\nQwen answer:")
    print(generated_answer)


    # --------------------------------------------------------
    # Convert llama.cpp response
    # --------------------------------------------------------

    try:

        artefactual_response = convert_llama_response(
            llama_result
        )

    except Exception as e:

        print(f"\nERROR converting response: {e}")

        results.append({
            "id": row["id"],
            "question": question,
            "reference_answer": reference_answer,
            "raw_response": raw_answer,
            "generated_answer": generated_answer,
            "non_hallucination_probability": None,
            "hallucination_probability": None,
            "error": f"Conversion error: {e}"
        })

        continue


    # --------------------------------------------------------
    # WEPR prediction
    # --------------------------------------------------------

    try:

        score = detector.predict_proba(
            artefactual_response
        )

        non_hallucination_probability = float(
            score[0][0]
        )

        hallucination_probability = float(
            score[0][1]
        )

    except Exception as e:

        print(f"\nERROR running WEPR: {e}")

        results.append({
            "id": row["id"],
            "question": question,
            "reference_answer": reference_answer,
            "raw_response": raw_answer,
            "generated_answer": generated_answer,
            "non_hallucination_probability": None,
            "hallucination_probability": None,
            "error": f"WEPR error: {e}"
        })

        continue


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print(
        f"WEPR hallucination probability: "
        f"{hallucination_probability:.4f}"
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append({

        "id": row["id"],

        "question": question,

        "reference_answer": reference_answer,

        "raw_response": raw_answer,

        "generated_answer": generated_answer,

        "non_hallucination_probability":
            non_hallucination_probability,

        "hallucination_probability":
            hallucination_probability,

        "error": ""

    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print(
    f"Successfully processed: "
    f"{results_df['hallucination_probability'].notna().sum()}"
)

print(
    f"Failed: "
    f"{results_df['hallucination_probability'].isna().sum()}"
)

print(
    f"\nResults saved to:\n"
    f"{OUTPUT_FILE}"
)