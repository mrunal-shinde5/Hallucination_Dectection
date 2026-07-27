import requests
import re
import torch

from typing import List, Dict, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# CONFIGURATION
# ============================================================

QWEN_URL = "http://127.0.0.1:8080/completion"

FALCON_URL = "http://127.0.0.1:8081/completion"

N_SAMPLES = 5

# Generate more candidates than needed.
# Bad perturbations will be filtered.
PERTURATION_CANDIDATES = 10

SELF_TEMPERATURE = 1.0

PERTURBED_TEMPERATURE = 0.0

PARAPHRASE_TEMPERATURE = 1.0

TIMEOUT = 180

# NLI threshold for question validation.
QUESTION_ENTAILMENT_THRESHOLD = 0.80

# NLI threshold for answer equivalence.
ANSWER_ENTAILMENT_THRESHOLD = 0.80


# ============================================================
# NLI MODEL
# ============================================================

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

print("\nLoading NLI model...")

nli_tokenizer = AutoTokenizer.from_pretrained(
    NLI_MODEL_NAME
)

nli_model = AutoModelForSequenceClassification.from_pretrained(
    NLI_MODEL_NAME
)

nli_model.eval()

print("NLI model loaded.")


# ============================================================
# GENERIC LOCAL MODEL REQUEST
# ============================================================

def local_generate(
    url: str,
    prompt: str,
    temperature: float = 0.0,
    n_predict: int = 100
) -> str:

    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "stream": False
    }

    response = requests.post(
        url,
        json=payload,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    result = response.json()

    return result.get(
        "content",
        ""
    ).strip()


# ============================================================
# QWEN
# ============================================================

def qwen_generate(
    prompt: str,
    temperature: float = 0.0,
    n_predict: int = 100
) -> str:

    return local_generate(
        QWEN_URL,
        prompt,
        temperature,
        n_predict
    )


# ============================================================
# FALCON
# ============================================================

def falcon_generate(
    prompt: str,
    temperature: float = 0.0,
    n_predict: int = 100
) -> str:

    return local_generate(
        FALCON_URL,
        prompt,
        temperature,
        n_predict
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    text: str
) -> str:

    if not text:
        return ""

    text = text.lower().strip()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# CLEAN ANSWER
# ============================================================

def clean_answer(
    answer: str
) -> str:

    if not answer:
        return ""

    answer = answer.strip()

    answer = re.sub(
        r"^(answer|response)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE
    )

    lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip()
    ]

    if lines:
        answer = lines[0]

    return answer.strip()


# ============================================================
# NLI ENTAILMENT PROBABILITY
# ============================================================

def entailment_probability(
    text_a: str,
    text_b: str
) -> float:

    if not text_a or not text_b:
        return 0.0

    if normalize_text(text_a) == normalize_text(text_b):
        return 1.0

    encoded = nli_tokenizer(
        text_a,
        text_b,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():

        logits = nli_model(
            **encoded
        ).logits

    probabilities = torch.softmax(
        logits,
        dim=-1
    )[0]

    # Find entailment label dynamically.
    entailment_index = None

    for idx, label in nli_model.config.id2label.items():

        if "entail" in label.lower():

            entailment_index = int(idx)

            break

    if entailment_index is None:

        raise RuntimeError(
            "Could not identify NLI entailment label."
        )

    return float(
        probabilities[
            entailment_index
        ]
    )


# ============================================================
# BIDIRECTIONAL SEMANTIC EQUIVALENCE
# ============================================================

def semantic_equivalence(
    text_a: str,
    text_b: str,
    threshold: float = ANSWER_ENTAILMENT_THRESHOLD
) -> Tuple[int, float]:

    if not text_a or not text_b:

        return 0, 0.0

    if (
        normalize_text(text_a)
        ==
        normalize_text(text_b)
    ):

        return 1, 1.0

    # A -> B
    p_ab = entailment_probability(
        text_a,
        text_b
    )

    # B -> A
    p_ba = entailment_probability(
        text_b,
        text_a
    )

    confidence = min(
        p_ab,
        p_ba
    )

    vote = int(
        p_ab >= threshold
        and
        p_ba >= threshold
    )

    return vote, confidence


# ============================================================
# QUESTION VALIDATION HELPERS
# ============================================================

def extract_numbers(
    text: str
) -> List[str]:

    return re.findall(
        r"\b\d+(?:\.\d+)?\b",
        text
    )


def extract_years(
    text: str
) -> List[str]:

    return re.findall(
        r"\b(?:1[0-9]{3}|20[0-9]{2})\b",
        text
    )


def contains_prompt_leakage(
    question: str
) -> bool:

    lower = question.lower()

    forbidden = [

        "original question",

        "generated questions",

        "equivalent questions",

        "here are the questions",

        "answer:",

        "let:"

    ]

    return any(
        phrase in lower
        for phrase in forbidden
    )


def looks_like_question(
    question: str
) -> bool:

    q = question.strip()

    if not q:
        return False

    if contains_prompt_leakage(q):
        return False

    question_starters = (

        "what ",
        "who ",
        "where ",
        "when ",
        "which ",
        "why ",
        "how ",
        "is ",
        "was ",
        "were ",
        "are ",
        "did ",
        "does ",
        "do ",
        "can ",
        "could ",
        "would ",
        "has ",
        "have "

    )

    return (
        "?" in q
        or q.lower().startswith(
            question_starters
        )
    )


# ============================================================
# VALIDATE PERTURBATION
# ============================================================

def validate_perturbation(
    original_question: str,
    candidate_question: str
) -> Tuple[bool, float, str]:

    candidate_question = (
        candidate_question.strip()
    )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    if not looks_like_question(
        candidate_question
    ):

        return (
            False,
            0.0,
            "Invalid question format"
        )

    # --------------------------------------------------------
    # Don't accept identical question
    # --------------------------------------------------------

    if (
        normalize_text(original_question)
        ==
        normalize_text(candidate_question)
    ):

        return (
            False,
            1.0,
            "Identical to original"
        )

    # --------------------------------------------------------
    # Preserve numbers
    #
    # If the original contains 1930, 1975, etc.,
    # the candidate should contain the same numbers.
    # --------------------------------------------------------

    original_numbers = extract_numbers(
        original_question
    )

    candidate_numbers = extract_numbers(
        candidate_question
    )

    if original_numbers != candidate_numbers:

        return (
            False,
            0.0,
            "Important numbers changed"
        )

    # --------------------------------------------------------
    # Bidirectional NLI
    #
    # Original -> Candidate
    # Candidate -> Original
    # --------------------------------------------------------

    p_original_to_candidate = (
        entailment_probability(
            original_question,
            candidate_question
        )
    )

    p_candidate_to_original = (
        entailment_probability(
            candidate_question,
            original_question
        )
    )

    confidence = min(
        p_original_to_candidate,
        p_candidate_to_original
    )

    valid = (
        p_original_to_candidate
        >= QUESTION_ENTAILMENT_THRESHOLD
        and
        p_candidate_to_original
        >= QUESTION_ENTAILMENT_THRESHOLD
    )

    if valid:

        reason = "Semantically equivalent"

    else:

        reason = "Semantic equivalence too low"

    return (
        valid,
        confidence,
        reason
    )


# ============================================================
# GENERATE + VALIDATE PERTURBATIONS
# ============================================================

def generate_valid_perturbations(
    original_question: str,
    required: int = N_SAMPLES
) -> Tuple[List[str], List[Dict]]:

    valid_questions = []

    validation_log = []

    attempts = 0

    max_attempts = 4

    while (
        len(valid_questions) < required
        and
        attempts < max_attempts
    ):

        attempts += 1

        print(
            f"\nPerturbation generation batch "
            f"{attempts}/{max_attempts}"
        )

        prompt = f"""
Generate {PERTURATION_CANDIDATES} different
semantically equivalent versions of the question below.

VERY IMPORTANT:

- Every question must ask EXACTLY the same thing.
- Preserve the same requested information.
- Preserve every person/entity.
- Preserve every date.
- Preserve every location.
- Preserve important numbers.
- Preserve the question's scope.
- If the original asks for a FIRST NAME, the rewrite
  must also ask for a FIRST NAME.
- If the original asks for a COUNTRY, the rewrite
  must also ask for a COUNTRY.
- Do not change first name into full name.
- Do not change a specific fact into a general fact.
- Do not add information.
- Do not remove information.
- Do not answer the question.
- Do not include explanations.
- Do not include "Original question".
- Do not include "Answer:".
- Put exactly one question on each line.

Original question:

{original_question}

Equivalent questions:
"""

        output = qwen_generate(
            prompt,
            temperature=PARAPHRASE_TEMPERATURE,
            n_predict=500
        )

        candidates = []

        for line in output.splitlines():

            line = line.strip()

            line = re.sub(
                r"^\s*\d+[\.\):\-]\s*",
                "",
                line
            )

            line = line.strip(
                "\"'"
            )

            if not line:
                continue

            candidates.append(
                line
            )

        for candidate in candidates:

            if len(valid_questions) >= required:
                break

            # Prevent duplicate candidates.
            if normalize_text(candidate) in [
                normalize_text(x)
                for x in valid_questions
            ]:
                continue

            valid, confidence, reason = (
                validate_perturbation(
                    original_question,
                    candidate
                )
            )

            validation_log.append({

                "candidate":
                    candidate,

                "valid":
                    valid,

                "confidence":
                    confidence,

                "reason":
                    reason

            })

            if valid:

                valid_questions.append(
                    candidate
                )

                print(
                    f"\nVALID perturbation "
                    f"{len(valid_questions)}:"
                )

                print(
                    candidate
                )

                print(
                    f"Confidence: "
                    f"{confidence:.4f}"
                )

            else:

                print(
                    "\nREJECTED perturbation:"
                )

                print(
                    candidate
                )

                print(
                    f"Reason: {reason}"
                )

    return (
        valid_questions[:required],
        validation_log
    )


# ============================================================
# SELF RESPONSES
# ============================================================

def generate_self_responses(
    question: str,
    number: int = N_SAMPLES
) -> List[str]:

    responses = []

    for i in range(number):

        print(
            f"Generating self-response "
            f"{i + 1}/{number}..."
        )

        prompt = f"""
Answer the following question.

Give only the answer.
Do not explain your reasoning.
Do not repeat the question.

Question:
{question}

Answer:
"""

        answer = qwen_generate(
            prompt,
            temperature=SELF_TEMPERATURE,
            n_predict=80
        )

        responses.append(
            clean_answer(answer)
        )

    return responses


# ============================================================
# PERTURBED RESPONSES
# ============================================================

def generate_perturbed_responses(
    questions: List[str]
) -> List[str]:

    responses = []

    for i, question in enumerate(
        questions,
        start=1
    ):

        print(
            f"Answering valid perturbed question "
            f"{i}/{len(questions)}..."
        )

        prompt = f"""
Answer the following question.

Give only the answer.
Do not explain your reasoning.
Do not repeat the question.

Question:
{question}

Answer:
"""

        answer = qwen_generate(
            prompt,
            temperature=PERTURBED_TEMPERATURE,
            n_predict=80
        )

        responses.append(
            clean_answer(answer)
        )

    return responses


# ============================================================
# SC²
# ============================================================

def calculate_sc2(
    target_answer: str,
    candidate_answers: List[str]
) -> Tuple[float, List[int], List[float]]:

    votes = []

    confidences = []

    for answer in candidate_answers:

        vote, confidence = (
            semantic_equivalence(
                target_answer,
                answer
            )
        )

        votes.append(
            vote
        )

        confidences.append(
            confidence
        )

    if not votes:

        return 0.0, [], []

    score = (
        sum(votes)
        /
        len(votes)
    )

    return (
        score,
        votes,
        confidences
    )


# ============================================================
# SAC³-Q
# ============================================================

def calculate_sac3_q(
    target_answer: str,
    perturbed_answers: List[str]
) -> Tuple[float, List[int], List[float]]:

    votes = []

    confidences = []

    for answer in perturbed_answers:

        vote, confidence = (
            semantic_equivalence(
                target_answer,
                answer
            )
        )

        votes.append(
            vote
        )

        confidences.append(
            confidence
        )

    if not votes:

        return 0.0, [], []

    score = (
        sum(votes)
        /
        len(votes)
    )

    return (
        score,
        votes,
        confidences
    )


# ============================================================
# FALCON ANSWER
# ============================================================

def generate_falcon_answer(
    question: str
) -> str:

    prompt = f"""
Answer the following question.

Give only the answer.
Do not explain your reasoning.
Do not repeat the question.

Question:
{question}

Answer:
"""

    answer = falcon_generate(
        prompt,
        temperature=0.0,
        n_predict=80
    )

    return clean_answer(
        answer
    )


# ============================================================
# CROSS-MODEL CONSISTENCY
# ============================================================

def calculate_cross_model_consistency(
    qwen_answer: str,
    falcon_answer: str
) -> Tuple[int, float]:

    return semantic_equivalence(
        qwen_answer,
        falcon_answer
    )


# ============================================================
# COMPLETE SAC³
# ============================================================

def run_sac3(
    question: str,
    target_answer: str,
    number: int = N_SAMPLES
) -> Dict:

    print(
        "\n" + "=" * 70
    )

    print(
        "SAC³ ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "\nQuestion:"
    )

    print(
        question
    )

    print(
        "\nTarget Qwen answer:"
    )

    print(
        target_answer
    )

    # ========================================================
    # PERTURBATIONS
    # ========================================================

    print(
        "\nGenerating and validating "
        "semantic perturbations..."
    )

    (
        perturbed_questions,
        validation_log
    ) = generate_valid_perturbations(
        question,
        required=number
    )

    if len(perturbed_questions) < number:

        print(
            "\nWARNING:"
        )

        print(
            f"Only {len(perturbed_questions)} "
            f"valid perturbations were generated."
        )

        print(
            "SAC³-Q will use the valid perturbations."
        )

    print(
        "\nFinal VALID perturbations:"
    )

    for i, q in enumerate(
        perturbed_questions,
        start=1
    ):

        print(
            f"{i}: {q}"
        )

    # ========================================================
    # SELF RESPONSES
    # ========================================================

    print(
        "\nGenerating self responses..."
    )

    self_responses = (
        generate_self_responses(
            question,
            number
        )
    )

    print(
        "\nSelf responses:"
    )

    for i, answer in enumerate(
        self_responses,
        start=1
    ):

        print(
            f"{i}: {answer}"
        )

    # ========================================================
    # PERTURBED RESPONSES
    # ========================================================

    print(
        "\nGenerating responses to "
        "validated perturbations..."
    )

    perturbed_responses = (
        generate_perturbed_responses(
            perturbed_questions
        )
    )

    print(
        "\nPerturbed responses:"
    )

    for i, answer in enumerate(
        perturbed_responses,
        start=1
    ):

        print(
            f"{i}: {answer}"
        )

    # ========================================================
    # SC²
    # ========================================================

    print(
        "\nCalculating SC²..."
    )

    (
        sc2_score,
        sc2_votes,
        sc2_confidences
    ) = calculate_sc2(
        target_answer,
        self_responses
    )

    # ========================================================
    # SAC³-Q
    # ========================================================

    print(
        "\nCalculating SAC³-Q..."
    )

    (
        sac3_q_score,
        sac3_q_votes,
        sac3_q_confidences
    ) = calculate_sac3_q(
        target_answer,
        perturbed_responses
    )

    # ========================================================
    # FALCON
    # ========================================================

    print(
        "\nGenerating independent Falcon answer..."
    )

    falcon_answer = generate_falcon_answer(
        question
    )

    print(
        "\nFalcon answer:"
    )

    print(
        falcon_answer
    )

    print(
        "\nCalculating cross-model consistency..."
    )

    (
        cross_model_vote,
        cross_model_confidence
    ) = calculate_cross_model_consistency(
        target_answer,
        falcon_answer
    )

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "question":
            question,

        "target_answer":
            target_answer,

        "perturbed_questions":
            perturbed_questions,

        "validation_log":
            validation_log,

        "self_responses":
            self_responses,

        "perturbed_responses":
            perturbed_responses,

        "sc2_score":
            sc2_score,

        "sc2_votes":
            sc2_votes,

        "sc2_confidences":
            sc2_confidences,

        "sac3_q_score":
            sac3_q_score,

        "sac3_q_votes":
            sac3_q_votes,

        "sac3_q_confidences":
            sac3_q_confidences,

        "falcon_answer":
            falcon_answer,

        "cross_model_vote":
            cross_model_vote,

        "cross_model_confidence":
            cross_model_confidence
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = (
        "What is Bruce Willis' real first name?"
    )

    # This is the Qwen answer being evaluated.
    #
    # It is NOT the TriviaQA ground truth.
    target_answer = "David"

    result = run_sac3(
        question,
        target_answer,
        number=5
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "SAC³ TEST RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "\nSC² score:"
    )

    print(
        result["sc2_score"]
    )

    print(
        "SC² votes:"
    )

    print(
        result["sc2_votes"]
    )

    print(
        "SC² confidences:"
    )

    print(
        result["sc2_confidences"]
    )

    print(
        "\nSAC³-Q score:"
    )

    print(
        result["sac3_q_score"]
    )

    print(
        "SAC³-Q votes:"
    )

    print(
        result["sac3_q_votes"]
    )

    print(
        "SAC³-Q confidences:"
    )

    print(
        result["sac3_q_confidences"]
    )

    print(
        "\nFalcon answer:"
    )

    print(
        result["falcon_answer"]
    )

    print(
        "\nCross-model consistency:"
    )

    print(
        result["cross_model_vote"]
    )

    print(
        "Cross-model confidence:"
    )

    print(
        result["cross_model_confidence"]
    )

    print(
        "\nValid perturbations:"
    )

    print(
        len(
            result["perturbed_questions"]
        )
    )

    print(
        "\nRejected/accepted perturbation log:"
    )

    for item in result["validation_log"]:

        status = (
            "ACCEPTED"
            if item["valid"]
            else
            "REJECTED"
        )

        print(
            f"\n[{status}] "
            f"{item['candidate']}"
        )

        print(
            f"Confidence: "
            f"{item['confidence']:.4f}"
        )

        print(
            f"Reason: "
            f"{item['reason']}"
        )