import pandas as pd
import requests
import json
import os

from llama_to_artefactual import convert_llama_response
from artefactual.scoring.base_detector import wepr


# -----------------------------
# Configuration
# -----------------------------

INPUT_FILE = "../data/triviaqa_100.csv"
OUTPUT_FILE = "../results/triviaqa_10_results.csv"

QWEN_URL = "http://127.0.0.1:8080/completion"

N_QUESTIONS = 10


# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv(INPUT_FILE)

df = df.head(N_QUESTIONS)

print(f"Loaded {len(df)} questions.")


# -----------------------------
# Load WEPR detector
# -----------------------------

print("\nLoading WEPR detector...")

detector = wepr(
    "chicham/artefactual-wepr-phi4",
    k=15
)

print("WEPR detector loaded.")


# -----------------------------
# Process questions
# -----------------------------

results = []


for index, row in df.iterrows():

    question = row["question"]
    reference_answer = row["reference_answer"]

    print("\n" + "=" * 70)
    print(f"Question {index + 1}/{len(df)}")
    print("=" * 70)

    print("Question:")
    print(question)

    # -------------------------
    # Ask Qwen
    # -------------------------

    prompt = (
        "Answer the following question with only the answer. "
        "Do not explain. Do not repeat the answer.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    payload = {
        "prompt": prompt,
        "n_predict": 30,
        "temperature": 0,
        "n_probs": 15,
        "stream": False
    }

    response = requests.post(
        QWEN_URL,
        json=payload
    )

    response.raise_for_status()

    llama_result = response.json()

    generated_answer = llama_result["content"].strip()

    print("\nQwen answer:")
    print(generated_answer)

    # -------------------------
    # Convert response
    # -------------------------

    artefactual_response = convert_llama_response(
        llama_result
    )

    # -------------------------
    # WEPR prediction
    # -------------------------

    score = detector.predict_proba(
        artefactual_response
    )

    non_hallucination_probability = float(score[0][0])
    hallucination_probability = float(score[0][1])

    print("\nWEPR:")
    print(
        f"Hallucination probability: "
        f"{hallucination_probability:.4f}"
    )

    # -------------------------
    # Save result
    # -------------------------

    results.append({
        "id": row["id"],
        "question": question,
        "reference_answer": reference_answer,
        "generated_answer": generated_answer,
        "non_hallucination_probability":
            non_hallucination_probability,
        "hallucination_probability":
            hallucination_probability
    })


# -----------------------------
# Save results
# -----------------------------

results_df = pd.DataFrame(results)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("Experiment complete.")
print("=" * 70)

print(f"Saved results to: {OUTPUT_FILE}")

print("\nResults:")
print(results_df)