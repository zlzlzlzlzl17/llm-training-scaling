from pathlib import Path
from typing import Any

import fasttext


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CLASSIFIER_DIR = _PROJECT_ROOT / "local-shared-data" / "classifiers"

_MODEL_PATHS = {
    "nsfw": _CLASSIFIER_DIR / "dolma_fasttext_nsfw_jigsaw_model.bin",
    "toxic": _CLASSIFIER_DIR / "dolma_fasttext_hatespeech_jigsaw_model.bin",
}

_MODELS: dict[str, Any] = {}


def _get_model(model_name: str) -> Any:
    """Load each fastText model only once."""
    if model_name not in _MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name not in _MODELS:
        model_path = _MODEL_PATHS[model_name]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Classifier model not found: {model_path}"
            )

        _MODELS[model_name] = fasttext.load_model(str(model_path))

    return _MODELS[model_name]


def _predict_raw(model_name: str, text: str) -> tuple[str, float]:
    """Return the raw fastText label and its probability."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # fastText expects one example on one physical line.
    cleaned_text = " ".join(text.split())

    if not cleaned_text:
        return "non", 0.0

    model = _get_model(model_name)
    labels, probabilities = model.predict(cleaned_text, k=1)

    label = labels[0].removeprefix("__label__").lower()
    score = float(probabilities[0])

    return label, score


def classify_nsfw(text: str) -> tuple[str, float]:
    """Classify a document as nsfw or non-nsfw."""
    raw_label, score = _predict_raw("nsfw", text)

    label = "non-nsfw" if "non" in raw_label else "nsfw"
    return label, score


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """Classify a document as toxic or non-toxic."""
    raw_label, score = _predict_raw("toxic", text)

    label = "non-toxic" if "non" in raw_label else "toxic"
    return label, score
