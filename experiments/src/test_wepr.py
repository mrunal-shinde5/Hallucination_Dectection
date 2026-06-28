import json
from pathlib import Path

from artefactual.scoring.base_detector import wepr


# --------------------------------------------------
# 1. Load example JSON
# --------------------------------------------------

data_path = (
    Path(__file__).parent.parent
    / "data"
    / "open_ai_responses_top15.json"
)

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)


# --------------------------------------------------
# 2. Inspect JSON
# --------------------------------------------------

print("Loaded JSON successfully!")
print("Type:", type(data))
print("Keys:", data.keys())


# --------------------------------------------------
# 3. Get responses
# --------------------------------------------------

responses = data["responses"]

print("\nNumber of responses:", len(responses))

for i, response in enumerate(responses):
    print(f"\nResponse {i + 1}")
    print("Type:", type(response))
    print("Keys:", response.keys())


# --------------------------------------------------
# 4. Load WEPR detector
# --------------------------------------------------

detector = wepr(
    "chicham/artefactual-wepr-falcon3",
    k=15
)

print("\nWEPR detector loaded successfully!")
print(detector)


# --------------------------------------------------
# 5. Calculate WEPR scores
# --------------------------------------------------

for i, response in enumerate(responses):

    score = detector.predict_proba(response)

    print("\n" + "=" * 60)
    print(f"RESPONSE {i + 1}")
    print("=" * 60)

    print("Model:", response["model"])
    print("Output:", response["output"])
    print("WEPR hallucination probability:", score)