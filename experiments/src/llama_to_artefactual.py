def convert_llama_response(llama_result):
    """
    Convert a llama.cpp completion response into the
    logprob structure expected by Artefactual.
    """

    completion_probs = llama_result["completion_probabilities"]

    logprobs = []

    for token_data in completion_probs:

        token_info = {
            "token": token_data["token"],
            "logprob": token_data["logprob"],
            "top_logprobs": token_data["top_logprobs"]
        }

        logprobs.append(token_info)

    return {
        "output": [
            {
                "content": [
                    {
                        "logprobs": logprobs
                    }
                ]
            }
        ]
    }