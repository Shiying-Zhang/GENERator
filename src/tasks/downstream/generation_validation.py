import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "src"))

from custom_dataset import (  # noqa: E402
    DEFAULT_CONDITION_TOKENS,
    normalize_sequence_text,
    split_condition_prefix,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SuppressSpecialTokensLogitsProcessor:
    def __init__(self, special_token_ids: List[int]):
        self.special_token_ids = special_token_ids

    def __call__(self, input_ids, scores):
        for token_id in self.special_token_ids:
            scores[:, token_id] = -float("inf")
        return scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple post-SFT generation validation for enhancer sequences"
    )
    parser.add_argument("--model_name", type=str, required=True, help="Saved model path")
    parser.add_argument("--parquet_path", type=str, required=True, help="Validation/test parquet path")
    parser.add_argument("--sequence_col", type=str, default="sequence", help="Sequence column name")
    parser.add_argument("--conditioned_input", action="store_true", help="Treat sequence_col as control-token-prefixed DNA input")
    parser.add_argument("--condition_tokens", type=str, default=",".join(DEFAULT_CONDITION_TOKENS), help="Comma-separated control tokens recognized when --conditioned_input is enabled")
    parser.add_argument("--condition_filter", type=str, default=None, help="Optional activity bucket or condition token filter, such as high or <sp2>")
    parser.add_argument("--output_dir", type=str, default="results/generation_validation", help="Directory to save validation outputs")
    parser.add_argument("--num_samples", type=int, default=128, help="Number of sequences to validate")
    parser.add_argument("--prompt_bp_length", type=int, default=120, help="Prompt length in base pairs; must be a multiple of 6")
    parser.add_argument("--continuation_bp_length", type=int, default=126, help="Generated continuation length in base pairs; must be a multiple of 6")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 for A100-class GPUs")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Sampling top-p")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting target")
    parser.add_argument("--wandb_project", type=str, default="GENERator-validation", help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    return parser.parse_args()


def parse_condition_tokens(raw_tokens: str) -> List[str]:
    tokens = [token.strip() for token in str(raw_tokens).split(",") if token.strip()]
    return tokens or list(DEFAULT_CONDITION_TOKENS)


def should_use_wandb(report_to: str) -> bool:
    targets = {s.strip().lower() for s in str(report_to).split(",")}
    return "wandb" in targets


def setup_wandb(args) -> Optional[Any]:
    if not should_use_wandb(args.report_to):
        logger.info("W&B disabled (report_to=%s).", args.report_to)
        return None

    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args),
        reinit=True,
    )
    return run


def resolve_dtype(args):
    if args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        logger.info("Using BF16 precision.")
        return torch.bfloat16
    if args.bf16:
        logger.warning("BF16 requested but unsupported; using FP32.")
    logger.info("Using FP32 precision.")
    return torch.float32


def load_model_and_tokenizer(args, dtype):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def _json_ready(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def prepare_samples(args, tokenizer) -> List[Dict[str, str]]:
    if args.prompt_bp_length % 6 != 0 or args.continuation_bp_length % 6 != 0:
        raise ValueError("prompt_bp_length and continuation_bp_length must both be multiples of 6.")

    df = pd.read_parquet(args.parquet_path)
    df = df[df[args.sequence_col].notna()].copy()
    df[args.sequence_col] = df[args.sequence_col].map(
        lambda value: normalize_sequence_text(
            text=value,
            uppercase=True,
            strip=True,
            conditioned_input=args.conditioned_input,
            condition_tokens=args.condition_tokens,
        )
    )

    if args.condition_filter is not None:
        if "activity_bucket" in df.columns:
            df = df[df["activity_bucket"].astype(str) == args.condition_filter]
        elif "condition_token" in df.columns:
            df = df[df["condition_token"].astype(str) == args.condition_filter]
        else:
            raise ValueError(
                "condition_filter was provided but the parquet does not contain 'activity_bucket' or 'condition_token'."
            )

    df = df.sample(n=min(args.num_samples, len(df)), random_state=args.seed).reset_index(drop=False)

    samples: List[Dict[str, Any]] = []
    usable_len = args.prompt_bp_length + args.continuation_bp_length
    for _, row in df.iterrows():
        raw_seq = row[args.sequence_col]
        condition_token = ""
        dna_seq = raw_seq
        if args.conditioned_input:
            condition_token, dna_seq = split_condition_prefix(
                raw_seq,
                condition_tokens=args.condition_tokens,
            )

        dna_seq = dna_seq[: len(dna_seq) - (len(dna_seq) % tokenizer.k)]
        if len(dna_seq) < usable_len:
            continue
        prompt_dna = dna_seq[: args.prompt_bp_length]
        target = dna_seq[
            args.prompt_bp_length : args.prompt_bp_length + args.continuation_bp_length
        ]
        sample = {
            "prompt": prompt_dna,
            "prompt_model_input": condition_token + prompt_dna,
            "target": target,
            "reference_sequence": prompt_dna + target,
            "source_index": int(row["index"]),
        }
        if condition_token:
            sample["condition_token"] = condition_token
        if "activity_bucket" in row:
            sample["activity_bucket"] = str(row["activity_bucket"])
        if "id" in row:
            sample["source_id"] = str(row["id"])
        if "label" in row:
            sample["source_label"] = _json_ready(row["label"])
        samples.append(sample)

    if not samples:
        raise ValueError("No usable sequences found. Check sequence lengths and validation settings.")
    return samples


def batch_generate(samples: List[Dict[str, str]], args, tokenizer, model, device) -> List[Dict[str, Any]]:
    new_tokens = args.continuation_bp_length // tokenizer.k
    special_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens)
    logits_processor = LogitsProcessorList(
        [SuppressSpecialTokensLogitsProcessor(special_token_ids)]
    )

    rows: List[Dict[str, Any]] = []
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        prompts = [tokenizer.bos_token + row["prompt_model_input"] for row in batch]
        inputs = tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=new_tokens,
                min_new_tokens=new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=logits_processor,
            )

        generated_suffixes = tokenizer.batch_decode(
            outputs[:, -new_tokens:],
            skip_special_tokens=True,
        )

        for row, generated in zip(batch, generated_suffixes):
            target = row["target"]
            same = sum(1 for a, b in zip(generated, target) if a == b)
            bp_accuracy = same / len(target) if target else 0.0
            rows.append(
                {
                    "prompt": row["prompt"],
                    "prompt_model_input": row["prompt_model_input"],
                    "target": target,
                    "reference_sequence": row["reference_sequence"],
                    "generated_sequence": row["prompt"] + generated,
                    "generated": generated,
                    "generated_bp_length": len(generated),
                    "target_bp_length": len(target),
                    "bp_accuracy": bp_accuracy,
                    "is_valid_dna": set(generated).issubset({"A", "T", "C", "G"}),
                    "exact_match": generated == target,
                    "condition_token": row.get("condition_token"),
                    "activity_bucket": row.get("activity_bucket"),
                    "source_index": row.get("source_index"),
                    "source_id": row.get("source_id"),
                    "source_label": row.get("source_label"),
                }
            )
    return rows


def save_outputs(args, rows: List[Dict[str, Any]], summary: Dict[str, Any]):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    details_path = output_dir / "generation_details.jsonl"
    with open(details_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_ready(row), ensure_ascii=False) + "\n")

    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(summary), f, indent=2, ensure_ascii=False)

    logger.info("Saved generation details to %s", details_path)
    logger.info("Saved generation summary to %s", summary_path)


def main():
    args = parse_args()
    args.condition_tokens = parse_condition_tokens(args.condition_tokens)
    torch.manual_seed(args.seed)

    run = setup_wandb(args)
    dtype = resolve_dtype(args)
    tokenizer, model, device = load_model_and_tokenizer(args, dtype)
    samples = prepare_samples(args, tokenizer)
    rows = batch_generate(samples, args, tokenizer, model, device)

    mean_bp_accuracy = sum(row["bp_accuracy"] for row in rows) / len(rows)
    exact_match_rate = sum(1 for row in rows if row["exact_match"]) / len(rows)
    valid_dna_rate = sum(1 for row in rows if row["is_valid_dna"]) / len(rows)
    unique_rate = len({row["generated"] for row in rows}) / len(rows)
    mean_generated_bp_length = sum(row["generated_bp_length"] for row in rows) / len(rows)

    summary = {
        "model_name": args.model_name,
        "parquet_path": args.parquet_path,
        "sequence_col": args.sequence_col,
        "conditioned_input": args.conditioned_input,
        "condition_filter": args.condition_filter,
        "num_samples": len(rows),
        "prompt_bp_length": args.prompt_bp_length,
        "continuation_bp_length": args.continuation_bp_length,
        "mean_bp_accuracy": mean_bp_accuracy,
        "exact_match_rate": exact_match_rate,
        "valid_dna_rate": valid_dna_rate,
        "unique_rate": unique_rate,
        "mean_generated_bp_length": mean_generated_bp_length,
        "attn_implementation": args.attn_implementation,
        "bf16": args.bf16,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "device": device,
    }

    save_outputs(args, rows, summary)

    logger.info("Generation validation summary: %s", summary)

    if run is not None:
        wandb.log(summary)
        preview_rows = rows[: min(20, len(rows))]
        table = wandb.Table(columns=["condition_token", "activity_bucket", "prompt", "target", "generated", "bp_accuracy", "exact_match"])
        for row in preview_rows:
            table.add_data(
                row.get("condition_token"),
                row.get("activity_bucket"),
                row["prompt"],
                row["target"],
                row["generated"],
                row["bp_accuracy"],
                row["exact_match"],
            )
        wandb.log({"generation_preview": table})
        run.finish()


if __name__ == "__main__":
    main()
