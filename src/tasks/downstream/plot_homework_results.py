import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


DEFAULT_BUCKET_ORDER = ["low", "mid", "high"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot homework results for the DeepSTARR / GENERator assignments"
    )
    parser.add_argument(
        "--task1_test_results",
        type=str,
        default="results/deepstarr_regression/test_results.json",
        help="Path to task 1 test_results.json",
    )
    parser.add_argument(
        "--task2_generation_summary",
        type=str,
        default="results/deepstarr_sft_valid/generation_summary.json",
        help="Path to task 2 generation_summary.json",
    )
    parser.add_argument(
        "--task3_generation_summary",
        type=str,
        default="results/deepstarr_sft_conditioned_valid/generation_summary.json",
        help="Path to task 3 generation_summary.json",
    )
    parser.add_argument(
        "--task3_scoring_summary",
        type=str,
        default="results/deepstarr_conditioned_scoring/scoring_summary.json",
        help="Path to task 3 scoring_summary.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="report/figures",
        help="Directory to save plots",
    )
    return parser.parse_args()


def load_json(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Missing file: {p}")
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def configure_matplotlib():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11


def plot_task1_regression_metrics(task1: Dict[str, Any], output_dir: Path) -> Optional[Path]:
    pearson_values = [
        task1.get("test_pearson_label_0"),
        task1.get("test_pearson_label_1"),
        task1.get("test_pearson"),
    ]
    r2_values = [
        task1.get("test_r2_label_0"),
        task1.get("test_r2_label_1"),
        task1.get("test_r2"),
    ]
    if any(value is None for value in pearson_values + r2_values):
        return None

    labels = ["Label 0", "Label 1", "Overall"]
    colors = ["#2C7FB8", "#41B6C4", "#253494"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, values, title, ylabel in [
        (axes[0], pearson_values, "Task 1 Pearson Correlation", "Pearson"),
        (axes[1], r2_values, "Task 1 R2 Score", "R2"),
    ]:
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_ylim(0.0, max(0.85, max(values) + 0.08))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    fig.suptitle("Homework 1: Enhancer Activity Prediction", y=1.02, fontsize=14)
    fig.tight_layout()
    output_path = output_dir / "task1_regression_metrics.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_generation_comparison(
    task2: Dict[str, Any], task3: Dict[str, Any], output_dir: Path
) -> Optional[Path]:
    metrics = [
        ("mean_bp_accuracy", "BP Accuracy"),
        ("valid_dna_rate", "Valid DNA Rate"),
        ("unique_rate", "Unique Rate"),
    ]
    if any(task2.get(key) is None or task3.get(key) is None for key, _ in metrics):
        return None

    labels = [label for _, label in metrics]
    task2_values = [task2[key] for key, _ in metrics]
    task3_values = [task3[key] for key, _ in metrics]
    x = list(range(len(labels)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        task2_values,
        width=width,
        color="#3182BD",
        label="Task 2 Unconditional",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        task3_values,
        width=width,
        color="#E6550D",
        label="Task 3 Conditioned",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("Value")
    ax.set_title("Task 2 vs Task 3: Generation Quality Comparison")
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    output_path = output_dir / "task23_generation_comparison.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _get_bucket_summary(task3_scoring: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
    if "by_activity_bucket" in task3_scoring:
        return task3_scoring["by_activity_bucket"]
    if "by_condition_token" in task3_scoring:
        return task3_scoring["by_condition_token"]
    return None


def _ordered_buckets(bucket_summary: Dict[str, Dict[str, Any]]) -> List[str]:
    present = list(bucket_summary.keys())
    ordered = [bucket for bucket in DEFAULT_BUCKET_ORDER if bucket in present]
    remainder = [bucket for bucket in present if bucket not in ordered]
    return ordered + sorted(remainder)


def plot_task3_bucket_prediction_sum(
    task3_scoring: Dict[str, Any], output_dir: Path
) -> Optional[Path]:
    bucket_summary = _get_bucket_summary(task3_scoring)
    if not bucket_summary:
        return None

    buckets = _ordered_buckets(bucket_summary)
    generated = [bucket_summary[b]["mean_generated_prediction_sum"] for b in buckets]
    reference = [bucket_summary[b]["mean_reference_prediction_sum"] for b in buckets]
    x = list(range(len(buckets)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        reference,
        width=width,
        color="#9ECAE1",
        label="Reference Sequence",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        generated,
        width=width,
        color="#08519C",
        label="Generated Sequence",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([bucket.capitalize() for bucket in buckets])
    ax.set_ylabel("Predictor Score Sum")
    ax.set_title("Task 3: Predictor Score by Activity Bucket")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            y = bar.get_height()
            offset = 0.03 if y >= 0 else -0.07
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + offset,
                f"{y:.3f}",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=9,
            )

    fig.tight_layout()
    output_path = output_dir / "task3_bucket_prediction_sum.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_task3_bucket_deltas(
    task3_scoring: Dict[str, Any], output_dir: Path
) -> Optional[Path]:
    bucket_summary = _get_bucket_summary(task3_scoring)
    if not bucket_summary:
        return None

    buckets = _ordered_buckets(bucket_summary)
    delta_l0 = [bucket_summary[b].get("mean_prediction_delta_label_0", 0.0) for b in buckets]
    delta_l1 = [bucket_summary[b].get("mean_prediction_delta_label_1", 0.0) for b in buckets]
    delta_sum = [bucket_summary[b].get("mean_prediction_delta_sum", 0.0) for b in buckets]
    positive_rate = [bucket_summary[b].get("positive_delta_rate", 0.0) for b in buckets]
    x = list(range(len(buckets)))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))

    bars_l0 = axes[0].bar(
        [i - width / 2 for i in x],
        delta_l0,
        width=width,
        color="#31A354",
        label="Delta Label 0",
    )
    bars_l1 = axes[0].bar(
        [i + width / 2 for i in x],
        delta_l1,
        width=width,
        color="#756BB1",
        label="Delta Label 1",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([bucket.capitalize() for bucket in buckets])
    axes[0].set_ylabel("Prediction Delta")
    axes[0].set_title("Task 3: Label-wise Delta by Bucket")
    axes[0].legend()

    for bars in [bars_l0, bars_l1]:
        for bar in bars:
            y = bar.get_height()
            offset = 0.03 if y >= 0 else -0.06
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                y + offset,
                f"{y:.3f}",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=8,
            )

    bars_sum = axes[1].bar(
        [i - width / 2 for i in x],
        delta_sum,
        width=width,
        color="#E6550D",
        label="Delta Sum",
    )
    bars_rate = axes[1].bar(
        [i + width / 2 for i in x],
        positive_rate,
        width=width,
        color="#FDD0A2",
        label="Positive Delta Rate",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([bucket.capitalize() for bucket in buckets])
    axes[1].set_ylabel("Value")
    axes[1].set_ylim(min(-0.25, min(delta_sum) - 0.08), max(1.0, max(positive_rate) + 0.1))
    axes[1].set_title("Task 3: Overall Shift by Bucket")
    axes[1].legend()

    for bars in [bars_sum, bars_rate]:
        for bar in bars:
            y = bar.get_height()
            offset = 0.03 if y >= 0 else -0.06
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                y + offset,
                f"{y:.3f}",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=8,
            )

    fig.tight_layout()
    output_path = output_dir / "task3_bucket_deltas.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_compact_snapshot(
    task1: Optional[Dict[str, Any]],
    task2: Optional[Dict[str, Any]],
    task3_gen: Optional[Dict[str, Any]],
    task3_score: Optional[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    snapshot = {
        "task1": task1,
        "task2_generation": task2,
        "task3_generation": task3_gen,
        "task3_scoring": task3_score,
    }
    output_path = output_dir / "metrics_snapshot.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    return output_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    task1 = load_json(args.task1_test_results)
    task2 = load_json(args.task2_generation_summary)
    task3_gen = load_json(args.task3_generation_summary)
    task3_score = load_json(args.task3_scoring_summary)

    generated_paths = []
    if task1 is not None:
        path = plot_task1_regression_metrics(task1, output_dir)
        if path is not None:
            generated_paths.append(path)
    if task2 is not None and task3_gen is not None:
        path = plot_generation_comparison(task2, task3_gen, output_dir)
        if path is not None:
            generated_paths.append(path)
    if task3_score is not None:
        path = plot_task3_bucket_prediction_sum(task3_score, output_dir)
        if path is not None:
            generated_paths.append(path)
        path = plot_task3_bucket_deltas(task3_score, output_dir)
        if path is not None:
            generated_paths.append(path)

    snapshot_path = save_compact_snapshot(task1, task2, task3_gen, task3_score, output_dir)
    generated_paths.append(snapshot_path)

    print("Generated files:")
    for path in generated_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
