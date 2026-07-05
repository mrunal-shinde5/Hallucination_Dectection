from datasets import load_dataset

dataset = load_dataset(
    "mandarjoshi/trivia_qa",
    "rc.nocontext",
    split="train"
)

print(dataset)
print(dataset[0])