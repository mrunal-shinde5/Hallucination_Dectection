import requests

from artefactual.scoring.base_detector import wepr
from llama_to_artefactual import convert_llama_response


# --------------------------------------------------
# 1. Ask Qwen a question
# --------------------------------------------------

url = "http://127.0.0.1:8080/completion"

payload = {
    "prompt": "Who was the first person to walk on Mars?\nAnswer with only the answer, no explanation.",
    "n_predict": 5,
    "temperature": 0,
    "n_probs": 15,
    "stream": False
}

response = requests.post(url, json=payload)

response.raise_for_status()

llama_result = response.json()


# --------------------------------------------------
# 2. Display answer
# --------------------------------------------------

print("Qwen answer:")
print(llama_result["content"])


# --------------------------------------------------
# 3. Convert to Artefactual format
# --------------------------------------------------

artefactual_response = convert_llama_response(llama_result)

print("\nConverted response successfully.")


# --------------------------------------------------
# 4. Load WEPR
# --------------------------------------------------

detector = wepr(
    "chicham/artefactual-wepr-phi4",
    k=15
)

print("\nWEPR detector loaded.")


# --------------------------------------------------
# 5. Run hallucination detection
# --------------------------------------------------

score = detector.predict_proba(artefactual_response)

print("\nWEPR result:")
print(score)