"""
=========================================================
Evaluation Metrics
=========================================================
"""

from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction
)

from rouge_score import rouge_scorer

from sentence_transformers import (
    SentenceTransformer,
    util
)

from bert_score import score as bert_score


# =====================================================
# LOAD MODELS (Load Once)
# =====================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

rouge = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)

smooth = SmoothingFunction().method1


# =====================================================
# EXACT MATCH
# =====================================================

def compute_exact_match(reference, prediction):
    """
    Exact Match (Case Insensitive)
    """

    return int(
        reference.strip().lower() ==
        prediction.strip().lower()
    )


# =====================================================
# BLEU
# =====================================================

def compute_bleu(reference, prediction):
    """
    BLEU Score
    """

    return sentence_bleu(
        [reference.split()],
        prediction.split(),
        smoothing_function=smooth
    )


# =====================================================
# ROUGE-L
# =====================================================

def compute_rouge(reference, prediction):
    """
    ROUGE-L F1
    """

    score = rouge.score(
        reference,
        prediction
    )

    return score["rougeL"].fmeasure


# =====================================================
# COSINE SIMILARITY
# =====================================================

def compute_cosine(reference, prediction):
    """
    Cosine Similarity
    """

    emb1 = embedding_model.encode(
        reference,
        convert_to_tensor=True
    )

    emb2 = embedding_model.encode(
        prediction,
        convert_to_tensor=True
    )

    return util.pytorch_cos_sim(
        emb1,
        emb2
    ).item()


# =====================================================
# SEMANTIC SIMILARITY
# =====================================================

def compute_semantic(reference, prediction):
    """
    Semantic Similarity
    (Same as embedding cosine similarity)
    """

    return compute_cosine(
        reference,
        prediction
    )


# =====================================================
# SEMANTIC ACCURACY
# =====================================================

def compute_semantic_accuracy(
    reference,
    prediction,
    threshold=0.80
):
    """
    Returns 1 if similarity >= threshold
    """

    similarity = compute_cosine(
        reference,
        prediction
    )

    return int(similarity >= threshold)


# =====================================================
# BERTScore
# =====================================================

def compute_bertscore(reference, prediction):
    """
    BERTScore F1
    """

    _, _, F1 = bert_score(

        [prediction],

        [reference],

        lang="en",

        verbose=False

    )

    return F1.item()