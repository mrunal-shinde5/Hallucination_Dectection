import requests
import json

url = "http://127.0.0.1:8080/completion"

payload = {
    "prompt": "What is the capital of France? Answer briefly.",
    "n_predict": 10,
    "temperature": 0,
    "n_probs": 15,
    "stream": False
}

response = requests.post(url, json=payload)

print("HTTP status:", response.status_code)

result = response.json()

print("\nGenerated answer:")
print(result.get("content"))

print("\nKeys returned:")
print(result.keys())

# llama.cpp stores token probabilities here
completion_probs = result.get("completion_probabilities", [])

print("\nNumber of generated tokens with probability information:",
      len(completion_probs))

if completion_probs:

    print("\nFirst generated token:")
    print(completion_probs[0])

    print("\nNumber of top logprobs for first token:")

    top_logprobs = completion_probs[0].get("top_logprobs", [])

    print(len(top_logprobs))

    print("\nTop logprobs:")
    print(json.dumps(top_logprobs, indent=2))