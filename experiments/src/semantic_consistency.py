import numpy as np
from sentence_transformers import CrossEncoder


# ------------------------------------------------------------
# NLI MODEL
# ------------------------------------------------------------

MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"

print("Loading NLI model...")

nli_model = CrossEncoder(MODEL_NAME)

print("NLI model loaded.")


# Label indices for this model:
# 0 = contradiction
# 1 = entailment
# 2 = neutral

ENTAILMENT_INDEX = 1


# ------------------------------------------------------------
# BIDIRECTIONAL SEMANTIC EQUIVALENCE
# ------------------------------------------------------------

def are_semantically_equivalent(
    answer_a,
    answer_b,
    threshold=0.5
):
    """
    Determine whether two answers express the same meaning.

    We check entailment in both directions:

        A -> B
        B -> A

    If both directions have sufficiently high
    entailment probability, we treat the answers
    as semantically equivalent.
    """

    pairs = [
        (answer_a, answer_b),
        (answer_b, answer_a)
    ]

    scores = nli_model.predict(pairs)

    entailment_ab = float(
        scores[0][ENTAILMENT_INDEX]
    )

    entailment_ba = float(
        scores[1][ENTAILMENT_INDEX]
    )

    equivalent = (
        entailment_ab >= threshold
        and entailment_ba >= threshold
    )

    return equivalent, entailment_ab, entailment_ba


# ------------------------------------------------------------
# CLUSTER ANSWERS BY MEANING
# ------------------------------------------------------------

def cluster_answers(
    answers,
    threshold=0.5
):
    """
    Group answers that are mutually semantically equivalent.

    A simple graph/connected-component approach is used:

        answer A <-> answer B
                 |
                 v
             same cluster
    """

    n = len(answers)

    # Initially each answer is its own cluster.
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]

        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    pair_results = []

    # Compare every pair.
    for i in range(n):

        for j in range(i + 1, n):

            equivalent, entail_ab, entail_ba = (
                are_semantically_equivalent(
                    answers[i],
                    answers[j],
                    threshold
                )
            )

            pair_results.append({
                "answer_a": i,
                "answer_b": j,
                "entailment_a_to_b": entail_ab,
                "entailment_b_to_a": entail_ba,
                "equivalent": equivalent
            })

            if equivalent:
                union(i, j)

    # Build clusters.
    clusters = {}

    for i in range(n):

        root = find(i)

        if root not in clusters:
            clusters[root] = []

        clusters[root].append(i)

    clusters = list(clusters.values())

    return clusters, pair_results


# ------------------------------------------------------------
# SEMANTIC ENTROPY
# ------------------------------------------------------------

def calculate_semantic_entropy(clusters, total_answers):
    """
    Calculate entropy over semantic meaning clusters.

    Higher entropy:
        More competing meanings
        -> lower semantic consistency

    Lower entropy:
        One dominant meaning
        -> higher semantic consistency
    """

    probabilities = []

    for cluster in clusters:

        probability = (
            len(cluster) / total_answers
        )

        probabilities.append(probability)

    entropy = 0.0

    for probability in probabilities:

        if probability > 0:

            entropy -= (
                probability *
                np.log(probability)
            )

    return float(entropy)


# ------------------------------------------------------------
# NORMALIZED SEMANTIC ENTROPY
# ------------------------------------------------------------

def calculate_semantic_consistency(
    answers,
    threshold=0.5
):
    """
    Complete semantic consistency calculation.

    Returns:
        clusters
        semantic entropy
        normalized entropy
        consistency score
    """

    if len(answers) < 2:

        return {
            "clusters": [[0]],
            "semantic_entropy": 0.0,
            "normalized_entropy": 0.0,
            "semantic_consistency": 1.0
        }

    clusters, pair_results = cluster_answers(
        answers,
        threshold
    )

    entropy = calculate_semantic_entropy(
        clusters,
        len(answers)
    )

    # Maximum entropy occurs when every answer
    # has a different meaning.
    max_entropy = np.log(len(answers))

    if max_entropy > 0:

        normalized_entropy = (
            entropy / max_entropy
        )

    else:

        normalized_entropy = 0.0

    # Convert entropy into consistency.
    #
    # 1 = highly consistent
    # 0 = highly inconsistent
    consistency = 1.0 - normalized_entropy

    return {
        "clusters": clusters,
        "pair_results": pair_results,
        "semantic_entropy": float(entropy),
        "normalized_entropy": float(
            normalized_entropy
        ),
        "semantic_consistency": float(
            consistency
        )
    }


# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------

if __name__ == "__main__":

    test_answers = [
        "Paris",
        "Paris",
        "The capital of France is Paris.",
        "Paris, France",
        "London"
    ]

    result = calculate_semantic_consistency(
        test_answers
    )

    print("\nTest answers:")

    for i, answer in enumerate(test_answers):

        print(
            f"{i}: {answer}"
        )

    print("\nClusters:")

    for cluster in result["clusters"]:

        print(cluster)

    print(
        "\nSemantic entropy:",
        result["semantic_entropy"]
    )

    print(
        "Normalized entropy:",
        result["normalized_entropy"]
    )

    print(
        "Semantic consistency:",
        result["semantic_consistency"]
    )