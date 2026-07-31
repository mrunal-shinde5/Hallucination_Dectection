from datasets import load_dataset
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

N_QUESTIONS = 500

RANDOM_SEED = 42

OUTPUT_PATH = "../data/triviaqa_500.csv"


# ============================================================
# LOAD TRIVIAQA
# ============================================================

print("=" * 70)
print("LOADING TRIVIAQA")
print("=" * 70)

dataset = load_dataset(
    "mandarjoshi/trivia_qa",
    "rc.nocontext",
    split="train"
)

print(
    f"\nTotal questions available: {len(dataset)}"
)


# ============================================================
# CHECK DATASET SIZE
# ============================================================

if len(dataset) < N_QUESTIONS:

    raise ValueError(
        f"Dataset contains only {len(dataset)} "
        f"questions, but {N_QUESTIONS} are required."
    )


# ============================================================
# RANDOMLY SAMPLE 500 QUESTIONS
# ============================================================

print(
    f"\nSelecting {N_QUESTIONS} questions..."
)

sample = dataset.shuffle(
    seed=RANDOM_SEED
).select(
    range(N_QUESTIONS)
)


# ============================================================
# EXTRACT REQUIRED INFORMATION
# ============================================================

data = []


for i, item in enumerate(sample):

    data.append({

        "id": i + 1,

        "question":
            item["question"],

        "reference_answer":
            item["answer"]["value"]

    })


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

df = pd.DataFrame(
    data
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "VALIDATING DATASET"
)

print(
    "=" * 70
)


# Check number of questions

assert len(df) == N_QUESTIONS


# Check duplicate questions

duplicate_questions = (
    df["question"]
    .duplicated()
    .sum()
)


print(
    f"\nDuplicate questions: "
    f"{duplicate_questions}"
)


if duplicate_questions > 0:

    raise ValueError(
        "Duplicate questions found."
    )


# Check missing questions

missing_questions = (
    df["question"]
    .isna()
    .sum()
)


# Check missing reference answers

missing_answers = (
    df["reference_answer"]
    .isna()
    .sum()
)


print(
    f"Missing questions: "
    f"{missing_questions}"
)

print(
    f"Missing reference answers: "
    f"{missing_answers}"
)


if missing_questions > 0:

    raise ValueError(
        "Missing questions found."
    )


if missing_answers > 0:

    raise ValueError(
        "Missing reference answers found."
    )


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "500-QUESTION DATASET CREATED"
)

print(
    "=" * 70
)

print(
    f"\nQuestions saved: "
    f"{len(df)}"
)

print(
    f"Random seed: "
    f"{RANDOM_SEED}"
)

print(
    f"Output:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# FIRST 10 QUESTIONS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FIRST 10 SELECTED QUESTIONS"
)

print(
    "=" * 70
)

print(
    df.head(10).to_string(
        index=False
    )
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "DATASET READY"
)

print(
    "=" * 70
)

print(
    "\nUse this exact file for:"
)

print(
    "E1 — WEPR"
)

print(
    "E2 — Semantic Consistency"
)

print(
    "E3 — WEPR + Semantic Hybrid"
)

print(
    "\nDo not generate a different sample "
    "for individual experiments."
)