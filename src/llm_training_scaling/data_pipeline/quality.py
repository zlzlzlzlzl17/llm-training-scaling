import re
from functools import lru_cache

import fasttext

from llm_training_scaling.data_pipeline.common import get_shared_assets_path


_WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def gopher_quality_filter(text: str) -> bool:
    """Return whether a document passes the required Gopher quality rules."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Extract word-like tokens while excluding standalone punctuation.
    words = _WORD_PATTERN.findall(text)
    num_words = len(words)

    # Rule 1: document length.
    if num_words < 50 or num_words > 100_000:
        return False

    # Rule 2: mean word length.
    mean_word_length = sum(len(word) for word in words) / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # Rule 3: proportion of lines ending in an ellipsis.
    lines = text.splitlines() or [text]
    ellipsis_lines = sum(line.rstrip().endswith("...") for line in lines)

    if ellipsis_lines / len(lines) > 0.30:
        return False

    # Rule 4: proportion of words containing at least one letter.
    alphabetic_words = sum(
        any(character.isalpha() for character in word)
        for word in words
    )

    if alphabetic_words / num_words < 0.80:
        return False

    return True

QUALITY_THRESHOLD = 0.45


@lru_cache(maxsize=1)
def _load_quality_classifier():
    model_path = (
        get_shared_assets_path()
        / "classifiers"
        / "quality_classifier.bin"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Quality classifier not found: {model_path}"
        )

    return fasttext.load_model(str(model_path))


def classify_quality(text: str) -> tuple[str, float]:
    """Classify text as wiki-like or ordinary Common Crawl text."""
    normalized_text = " ".join(text.split())

    if not normalized_text:
        return "cc", 0.0

    model = _load_quality_classifier()

    # Request both labels so that we can explicitly inspect P(wiki).
    labels, scores = model.predict(
        normalized_text,
        k=2,
    )

    probabilities = {
        label.removeprefix("__label__"): float(score)
        for label, score in zip(labels, scores)
    }

    wiki_score = probabilities.get("wiki", 0.0)
    cc_score = probabilities.get("cc", 0.0)

    if wiki_score >= QUALITY_THRESHOLD:
        return "wiki", wiki_score

    return "cc", cc_score