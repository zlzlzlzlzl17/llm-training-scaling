from pathlib import Path
from typing import Any

import fasttext


_MODEL: Any | None = None


def _get_language_model() -> Any:
    """Load the fastText model once and reuse it."""
    global _MODEL

    if _MODEL is None:
        project_root = Path(__file__).resolve().parents[1]
        model_path = (
            project_root
            / "local-shared-data"
            / "classifiers"
            / "lid.176.bin"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Language identification model not found: {model_path}"
            )

        _MODEL = fasttext.load_model(str(model_path))

    return _MODEL


def identify_language(text: str) -> tuple[str, float]:
    """Return the most likely language code and its confidence."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # fastText expects each input example to be on one physical line.
    cleaned_text = " ".join(text.split())

    if not cleaned_text:
        return "unknown", 0.0

    model = _get_language_model()
    labels, probabilities = model.predict(cleaned_text, k=1)

    label = labels[0]
    confidence = float(probabilities[0])

    # "__label__en" -> "en"
    language = label.removeprefix("__label__")

    return language, confidence
