from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

# Allow direct execution from the repository root without an editable install.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llm_training_scaling.alignment.rewards import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)
from llm_training_scaling.alignment.rollout_server import VLLMServer


PromptConfig = tuple[Path, Callable[[str, str], dict[str, float]], bool]

PROMPT_CONFIGS: dict[str, PromptConfig] = {
    "question_only": (
        ROOT / "src/llm_training_scaling/alignment/prompts/question_only.prompt",
        question_only_reward_fn,
        False,
    ),
    "r1_zero": (
        ROOT / "src/llm_training_scaling/alignment/prompts/r1_zero.prompt",
        r1_zero_reward_fn,
        True,
    ),
    "r1_zero_three_shot": (
        ROOT / "src/llm_training_scaling/alignment/prompts/r1_zero_three_shot_gsm8k.prompt",
        r1_zero_reward_fn,
        True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-type",
        choices=["all", *PROMPT_CONFIGS.keys()],
        default="all",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "data/gsm8k/test.jsonl",
    )
    parser.add_argument("--model-id", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/prompting",
    )
    return parser.parse_args()


def load_examples(path: Path, limit: int) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(examples) >= limit:
                break

            row = json.loads(line)
            examples.append(
                {
                    "question": row["question"],
                    "ground_truth": row["answer"].split("####")[-1].strip(),
                }
            )

    if not examples:
        raise ValueError(f"No examples were loaded from {path}")

    return examples


def build_prompts(template: str, examples: list[dict[str, str]]) -> list[str]:
    # 用 replace 避免 prompt 中其他大括號被 str.format 誤解
    return [
        template.replace("{question}", example["question"])
        for example in examples
    ]


def classify_reward(format_reward: float, answer_reward: float) -> str:
    if format_reward == 1 and answer_reward == 1:
        return "format_1_answer_1"
    if format_reward == 1 and answer_reward == 0:
        return "format_1_answer_0"
    if format_reward == 0 and answer_reward == 0:
        return "format_0_answer_0"
    return f"format_{format_reward}_answer_{answer_reward}"


def evaluate_prompt_type(
    server: VLLMServer,
    prompt_type: str,
    examples: list[dict[str, str]],
    batch_size: int,
    output_dir: Path,
) -> dict[str, object]:
    template_path, reward_fn, uses_answer_tags = PROMPT_CONFIGS[prompt_type]
    template = template_path.read_text(encoding="utf-8")
    prompts = build_prompts(template, examples)

    sampling_params: dict[str, object] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "n": 1,
        "seed": 0,
    }

    if uses_answer_tags:
        sampling_params["stop"] = ["</answer>"]
        sampling_params["include_stop_str_in_output"] = True

    print(f"\n{'=' * 80}")
    print(f"Evaluating: {prompt_type}")
    print(f"Examples:   {len(examples)}")
    print(f"{'=' * 80}")

    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )

    if len(completions) != len(examples):
        raise RuntimeError(
            f"Expected {len(examples)} completions, got {len(completions)}"
        )

    output_path = output_dir / f"{prompt_type}.jsonl"
    category_counts: Counter[str] = Counter()
    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0
    total_tokens = 0

    with output_path.open("w", encoding="utf-8") as f:
        for index, (example, prompt, completion) in enumerate(
            zip(examples, prompts, completions, strict=True)
        ):
            reward_result = reward_fn(
                completion.text,
                example["ground_truth"],
            )

            reward = float(reward_result["reward"])
            format_reward = float(reward_result["format_reward"])
            answer_reward = float(reward_result["answer_reward"])
            category = classify_reward(format_reward, answer_reward)

            category_counts[category] += 1
            total_reward += reward
            total_format_reward += format_reward
            total_answer_reward += answer_reward
            total_tokens += len(completion.token_ids)

            record = {
                "index": index,
                "prompt_type": prompt_type,
                "question": example["question"],
                "ground_truth": example["ground_truth"],
                "prompt": prompt,
                "response": completion.text,
                "finish_reason": completion.finish_reason,
                "generated_tokens": len(completion.token_ids),
                "reward": reward,
                "format_reward": format_reward,
                "answer_reward": answer_reward,
                "category": category,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n = len(examples)
    summary: dict[str, object] = {
        "prompt_type": prompt_type,
        "num_examples": n,
        "mean_reward": total_reward / n,
        "answer_accuracy": total_answer_reward / n,
        "format_accuracy": total_format_reward / n,
        "average_generated_tokens": total_tokens / n,
        "categories": dict(category_counts),
        "output_path": str(output_path),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.data_path, args.limit)

    selected_prompt_types = (
        list(PROMPT_CONFIGS)
        if args.prompt_type == "all"
        else [args.prompt_type]
    )

    print(f"Starting vLLM with model: {args.model_id}")

    server = VLLMServer(
        model_id=args.model_id,
        gpu=args.gpu,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        startup_timeout=1800,
        logging_level="INFO",
    )
    server.start()

    summaries = []
    for prompt_type in selected_prompt_types:
        summaries.append(
            evaluate_prompt_type(
                server=server,
                prompt_type=prompt_type,
                examples=examples,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
            )
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
