from datasets import load_dataset
import pandas as pd


# Load TriviaQA
dataset = load_dataset(
    "mandarjoshi/trivia_qa",
    "rc.nocontext",
    split="train"
)

# Select first 100 questions
sample = dataset.select(range(100))

# Extract the information we need
data = []

for i, item in enumerate(sample):
    data.append({
        "id": i + 1,
        "question": item["question"],
        "reference_answer": item["answer"]["value"],
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save
output_path = "../data/triviaqa_100.csv"
df.to_csv(output_path, index=False)

print(f"Saved {len(df)} questions to {output_path}")
print("\nFirst 5 questions:")
print(df.head())