# nmt_backend.py
# Neural Machine Translation Backend with BLEU Evaluation

import torch
import math
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from collections import Counter

# -------------------------
# LOAD MODEL (English → Hindi)
# -------------------------
MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"

print("Loading translation model...")
try:
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    print(f"Model '{MODEL_NAME}' loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise


# -------------------------
# HELPER FUNCTIONS FOR N-GRAMS
# -------------------------
def get_ngrams(tokens, n):
    """Extract n-grams from token list"""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def count_ngrams(tokens_list, n):
    """Count n-grams in list of token lists"""
    counts = Counter()
    for tokens in tokens_list:
        counts.update(get_ngrams(tokens, n))
    return counts


# -------------------------
# TRANSLATION FUNCTION
# -------------------------
def generate_translations(
        text,
        num_beams=5,
        num_candidates=3,
        max_length=256
):
    """Generate multiple candidate translations using beam search"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_candidates,
            max_length=max_length,
            early_stopping=True,
            length_penalty=1.0,  # Length normalization
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
        )

    translations = [
        tokenizer.decode(t, skip_special_tokens=True).strip()
        for t in outputs
    ]

    # Remove duplicates while preserving order
    unique_translations = []
    seen = set()
    for t in translations:
        if t not in seen:
            seen.add(t)
            unique_translations.append(t)

    return unique_translations[:num_candidates]  # Return requested number


# -------------------------
# BLEU SCORE WITH BREVITY PENALTY (EXPLICIT IMPLEMENTATION)
# -------------------------
def compute_bleu_with_brevity(candidate, references):
    """
    Compute BLEU score with explicit brevity penalty calculation
    Returns: (bleu_score, ngram_precisions, brevity_penalty)
    """
    # Tokenize
    cand_tokens = candidate.split()
    ref_tokens_list = [ref.split() for ref in references]

    # Calculate reference lengths
    ref_lengths = [len(ref_tokens) for ref_tokens in ref_tokens_list]
    cand_length = len(cand_tokens)

    # 1. FIND CLOSEST REFERENCE LENGTH
    closest_ref_len = min(ref_lengths, key=lambda x: abs(x - cand_length))

    # 2. CALCULATE BREVITY PENALTY
    if cand_length == 0:
        return 0.0, {f"{n}-gram": 0.0 for n in range(1, 5)}, 0.0

    if cand_length < closest_ref_len:
        brevity_penalty = math.exp(1 - closest_ref_len / cand_length)
    else:
        brevity_penalty = 1.0

    # 3. CALCULATE MODIFIED N-GRAM PRECISIONS
    ngram_precisions = {}

    for n in range(1, 5):  # 1-gram to 4-gram
        cand_ngrams = get_ngrams(cand_tokens, n)

        if not cand_ngrams:  # Candidate shorter than n
            ngram_precisions[n] = 0.0
            continue

        # Count max n-gram occurrences in references
        max_counts = {}
        for ref_tokens in ref_tokens_list:
            ref_ngrams = get_ngrams(ref_tokens, n)
            ref_counts = Counter(ref_ngrams)

            for ngram in cand_ngrams:
                if ngram in ref_counts:
                    max_counts[ngram] = max(max_counts.get(ngram, 0), ref_counts[ngram])
                else:
                    max_counts[ngram] = max_counts.get(ngram, 0)  # Keep existing or 0

        # Calculate modified precision
        clip_count = 0
        for ngram in cand_ngrams:
            clip_count += min(cand_ngrams.count(ngram), max_counts.get(ngram, 0))

        ngram_precisions[n] = clip_count / len(cand_ngrams)

    # 4. GEOMETRIC MEAN OF PRECISIONS
    weights = [0.25, 0.25, 0.25, 0.25]  # Standard BLEU weights
    precision_product = 1.0

    for n in range(1, 5):
        if ngram_precisions[n] > 0:
            precision_product *= ngram_precisions[n] ** weights[n - 1]
        else:
            precision_product = 0.0
            break

    # 5. FINAL BLEU SCORE
    bleu_score = brevity_penalty * precision_product

    # Format n-gram precisions for display
    ngram_display = {
        f"{n}-gram": ngram_precisions[n] for n in range(1, 5)
    }

    return bleu_score, ngram_display, brevity_penalty


# -------------------------
# ALTERNATIVE: USING NLTK (for comparison/backup)
# -------------------------
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


    def compute_bleu_nltk(candidate, references):
        """Compute BLEU using NLTK (for comparison)"""
        smoothie = SmoothingFunction().method4
        ref_tokens = [ref.split() for ref in references]
        cand_tokens = candidate.split()

        bleu_score = sentence_bleu(
            ref_tokens,
            cand_tokens,
            smoothing_function=smoothie
        )

        # Individual n-gram scores
        ngram_scores = {
            "1-gram": sentence_bleu(ref_tokens, cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie),
            "2-gram": sentence_bleu(ref_tokens, cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
            "3-gram": sentence_bleu(ref_tokens, cand_tokens, weights=(0.33, 0.33, 0.33, 0),
                                    smoothing_function=smoothie),
            "4-gram": sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                                    smoothing_function=smoothie),
        }

        return bleu_score, ngram_scores
except ImportError:
    print("NLTK not available, using custom BLEU implementation")


# -------------------------
# FULL PIPELINE
# -------------------------
def translate_and_evaluate(
        source_text,
        reference_texts,
        num_beams=5,
        num_candidates=3
):
    """
    Main pipeline: Translate source text and evaluate against references
    Returns: (results, best_index, brevity_penalties)
    """
    # Generate translations
    translations = generate_translations(
        source_text,
        num_beams=num_beams,
        num_candidates=num_candidates
    )

    # Evaluate each translation
    results = []
    brevity_penalties = []

    for t in translations:
        # Use custom BLEU with explicit brevity penalty
        bleu, ngrams, bp = compute_bleu_with_brevity(t, reference_texts)

        results.append({
            "translation": t,
            "bleu": bleu,
            "ngrams": ngrams
        })
        brevity_penalties.append(bp)

    # Find best translation
    if results:
        best_index = max(range(len(results)), key=lambda i: results[i]["bleu"])
    else:
        best_index = 0

    return results, best_index, brevity_penalties


# -------------------------
# TEST FUNCTION
# -------------------------
if __name__ == "__main__":
    # Test the pipeline
    test_source = "The algorithm uses dynamic programming for optimization."
    test_references = [
        "एल्गोरिथ्म अनुकूलन के लिए गतिशील प्रोग्रामिंग का उपयोग करता है।",
        "अनुकूलन हेतु एल्गोरिथ्म डायनामिक प्रोग्रामिंग प्रयोग करती है।"
    ]

    print("Testing translation pipeline...")
    print(f"Source: {test_source}")
    print(f"References: {test_references}")

    results, best_idx, bps = translate_and_evaluate(
        test_source,
        test_references,
        num_beams=5,
        num_candidates=3
    )

    print(f"\nGenerated {len(results)} candidates:")
    for i, r in enumerate(results):
        print(f"\nCandidate {i + 1} {'(BEST)' if i == best_idx else ''}:")
        print(f"  Translation: {r['translation']}")
        print(f"  BLEU: {r['bleu']:.4f}")
        print(f"  Brevity Penalty: {bps[i]:.4f}")
        for ngram, score in r['ngrams'].items():
            print(f"  {ngram}: {score:.4f}")