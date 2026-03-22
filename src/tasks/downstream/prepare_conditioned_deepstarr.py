import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BUCKET_TOKENS = {
    "low": "<sp0>",
    "mid": "<sp1>",
    "high": "<sp2>",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare bucket-conditioned DeepSTARR parquet files for controllable generation"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing DeepSTARR parquet splits (train/valid/test)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write conditioned parquet splits",
    )
    parser.add_argument(
        "--sequence_col",
        type=str,
        default="sequence",
        help="Sequence column name",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Label column containing a 2D activity vector",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Filename stem for the training split",
    )
    parser.add_argument(
        "--valid_split",
        type=str,
        default="valid",
        help="Filename stem for the validation split",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        help="Filename stem for the test split",
    )
    parser.add_argument(
        "--score_mode",
        type=str,
        default="sum",
        choices=["sum", "mean", "label_0", "label_1", "max"],
        help="How to collapse the 2D DeepSTARR label into a scalar activity score",
    )
    parser.add_argument(
        "--low_quantile",
        type=float,
        default=0.25,
        help="Quantile threshold for the low bucket",
    )
    parser.add_argument(
        "--high_quantile",
        type=float,
        default=0.75,
        help="Quantile threshold for the high bucket",
    )
    parser.add_argument(
        "--trim_multiple",
        type=int,
        default=6,
        help="Trim DNA sequences on the right to a multiple of this value",
    )
    parser.add_argument(
        "--metadata_name",
        type=str,
        default="conditioning_metadata.json",
        help="Metadata JSON filename written under output_dir",
    )
    return parser.parse_args()


def validate_args(args) -> None:
    if not 0.0 < args.low_quantile < args.high_quantile < 1.0:
        raise ValueError("Require 0 < low_quantile < high_quantile < 1.")
    if args.trim_multiple <= 0:
        raise ValueError("trim_multiple must be a positive integer.")


def read_split(path: Path) -> pd.DataFrame:
    logger.info("Loading %s", path)
    return pd.read_parquet(path)


def normalize_label(label: Any) -> List[float]:
    if isinstance(label, np.ndarray):
        return [float(x) for x in label.tolist()]
    if isinstance(label, (list, tuple)):
        return [float(x) for x in label]
    raise TypeError(f"Unsupported label type: {type(label)}")


def compute_activity_score(label: Any, score_mode: str) -> float:
    values = normalize_label(label)
    if len(values) < 2:
        raise ValueError(f"Expected at least 2 label dimensions, got {values}")

    if score_mode == "sum":
        return float(sum(values))
    if score_mode == "mean":
        return float(sum(values) / len(values))
    if score_mode == "label_0":
        return float(values[0])
    if score_mode == "label_1":
        return float(values[1])
    if score_mode == "max":
        return float(max(values))
    raise ValueError(f"Unknown score_mode: {score_mode}")


def trim_sequence(sequence: str, multiple: int) -> str:
    sequence = str(sequence).strip().upper()
    usable_length = len(sequence) - (len(sequence) % multiple)
    return sequence[:usable_length]


def assign_bucket(score: float, low_threshold: float, high_threshold: float) -> str:
    if score <= low_threshold:
        return "low"
    if score >= high_threshold:
        return "high"
    return "mid"


def bucket_counts(values: Iterable[str]) -> Dict[str, int]:
    counts = {"low": 0, "mid": 0, "high": 0}
    for value in values:
        counts[value] += 1
    return counts


def enrich_split(
    df: pd.DataFrame,
    split_name: str,
    args,
    low_threshold: float,
    high_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if args.sequence_col not in df.columns:
        raise KeyError(f"Missing sequence column '{args.sequence_col}' in split '{split_name}'.")
    if args.label_col not in df.columns:
        raise KeyError(f"Missing label column '{args.label_col}' in split '{split_name}'.")

    out = df.copy()
    out["activity_score"] = out[args.label_col].map(
        lambda x: compute_activity_score(x, args.score_mode)
    )
    out["trimmed_sequence"] = out[args.sequence_col].map(
        lambda x: trim_sequence(x, args.trim_multiple)
    )
    out["trimmed_sequence_bp_length"] = out["trimmed_sequence"].str.len()
    out["activity_bucket"] = out["activity_score"].map(
        lambda x: assign_bucket(x, low_threshold, high_threshold)
    )
    out["condition_token"] = out["activity_bucket"].map(BUCKET_TOKENS)
    out["condition_id"] = out["activity_bucket"].map({"low": 0, "mid": 1, "high": 2})
    out["conditioned_sequence"] = out["condition_token"] + out["trimmed_sequence"]
    out["source_split"] = split_name

    split_metadata = {
        "rows": int(len(out)),
        "bucket_counts": bucket_counts(out["activity_bucket"].tolist()),
        "activity_score_mean": float(out["activity_score"].mean()),
        "activity_score_std": float(out["activity_score"].std(ddof=0)),
        "trimmed_sequence_bp_length_min": int(out["trimmed_sequence_bp_length"].min()),
        "trimmed_sequence_bp_length_median": float(out["trimmed_sequence_bp_length"].median()),
        "trimmed_sequence_bp_length_max": int(out["trimmed_sequence_bp_length"].max()),
    }
    return out, split_metadata


def write_outputs(
    processed_splits: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any],
    output_dir: Path,
    metadata_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, df in processed_splits.items():
        output_path = output_dir / f"{split_name}.parquet"
        logger.info("Writing conditioned split to %s", output_path)
        df.to_parquet(output_path, index=False)

    metadata_path = output_dir / metadata_name
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Wrote metadata to %s", metadata_path)


def main():
    args = parse_args()
    validate_args(args)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    split_paths = {
        "train": input_dir / f"{args.train_split}.parquet",
        "valid": input_dir / f"{args.valid_split}.parquet",
        "test": input_dir / f"{args.test_split}.parquet",
    }

    if not split_paths["train"].exists():
        raise FileNotFoundError(f"Training split not found: {split_paths['train']}")

    available_splits = {
        split_name: path for split_name, path in split_paths.items() if path.exists()
    }

    raw_splits = {
        split_name: read_split(path) for split_name, path in available_splits.items()
    }

    train_scores = raw_splits["train"][args.label_col].map(
        lambda x: compute_activity_score(x, args.score_mode)
    )
    low_threshold = float(train_scores.quantile(args.low_quantile))
    high_threshold = float(train_scores.quantile(args.high_quantile))

    processed_splits: Dict[str, pd.DataFrame] = {}
    split_metadata: Dict[str, Any] = {}
    for split_name, df in raw_splits.items():
        processed_df, split_info = enrich_split(
            df=df,
            split_name=split_name,
            args=args,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        processed_splits[split_name] = processed_df
        split_metadata[split_name] = split_info

    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "score_mode": args.score_mode,
        "label_col": args.label_col,
        "sequence_col": args.sequence_col,
        "trim_multiple": args.trim_multiple,
        "bucket_tokens": BUCKET_TOKENS,
        "low_quantile": args.low_quantile,
        "high_quantile": args.high_quantile,
        "thresholds": {
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        },
        "splits": split_metadata,
        "notes": [
            "Thresholds are fit on the train split and then applied to valid/test.",
            "conditioned_sequence concatenates the condition token and right-trimmed DNA sequence.",
            "trimmed_sequence is made a multiple of trim_multiple for later GENERator tokenization.",
        ],
    }

    write_outputs(
        processed_splits=processed_splits,
        metadata=metadata,
        output_dir=output_dir,
        metadata_name=args.metadata_name,
    )


if __name__ == "__main__":
    main()
