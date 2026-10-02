from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_PATTERN = re.compile(
    r"iter\s+(\d+)/\d+\s+\|\s+"
    r"train loss\s+([0-9.eE+-]+)"
)

VALID_PATTERN = re.compile(
    r"iter\s+(\d+)\s+\|\s+"
    r"validation loss\s+([0-9.eE+-]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log",
        type=Path,
        default=Path("runs/tinystories_15m.log"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/tinystories_15m/analysis"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.log.exists():
        raise FileNotFoundError(
            f"Training log not found: {args.log}"
        )

    text = args.log.read_text(
        encoding="utf-8",
        errors="replace",
    )

    train_steps: list[int] = []
    train_losses: list[float] = []

    valid_steps: list[int] = []
    valid_losses: list[float] = []

    for match in TRAIN_PATTERN.finditer(text):
        train_steps.append(int(match.group(1)))
        train_losses.append(float(match.group(2)))

    for match in VALID_PATTERN.finditer(text):
        valid_steps.append(int(match.group(1)))
        valid_losses.append(float(match.group(2)))

    if not train_steps:
        raise RuntimeError(
            "No training-loss entries were found in the log."
        )

    if not valid_steps:
        raise RuntimeError(
            "No validation-loss entries were found in the log."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Train 与 validation loss 曲线
    plt.figure(figsize=(9, 5))
    plt.plot(
        train_steps,
        train_losses,
        label="Train loss",
        linewidth=1,
        alpha=0.65,
    )
    plt.plot(
        valid_steps,
        valid_losses,
        label="Validation loss",
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Optimizer step")
    plt.ylabel("Cross-entropy loss")
    plt.title("TinyStories Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    curve_path = args.output_dir / "loss_curve.png"
    plt.savefig(curve_path, dpi=160)
    plt.close()

    final_validation_loss = valid_losses[-1]
    final_perplexity = math.exp(final_validation_loss)

    best_index = min(
        range(len(valid_losses)),
        key=valid_losses.__getitem__,
    )

    summary = {
        "train_log_entries": len(train_losses),
        "validation_log_entries": len(valid_losses),
        "final_train_step": train_steps[-1],
        "final_train_loss": train_losses[-1],
        "final_validation_step": valid_steps[-1],
        "final_validation_loss": final_validation_loss,
        "final_perplexity": final_perplexity,
        "best_validation_step": valid_steps[best_index],
        "best_validation_loss": valid_losses[best_index],
        "best_validation_perplexity": math.exp(
            valid_losses[best_index]
        ),
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Training summary")
    print("=" * 60)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("=" * 60)
    print(f"Curve:   {curve_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
