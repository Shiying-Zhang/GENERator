import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import wandb
from transformers import AutoModelForSequenceClassification, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score generated enhancer sequences with a trained predictor"
    )
    parser.add_argument("--predictor_model", type=str, required=True, help="Path to sequence-understanding best_model")
    parser.add_argument("--generation_details_path", type=str, required=True, help="Path to generation_details.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/predictor_scoring", help="Directory to save scoring outputs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for scoring")
    parser.add_argument("--max_length", type=int, default=256, help="Max token length for predictor tokenization")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 scoring when supported")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting target")
    parser.add_argument("--wandb_project", type=str, default="GENERator-scoring", help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    return parser.parse_args()


def should_use_wandb(report_to: str) -> bool:
    targets = {s.strip().lower() for s in str(report_to).split(",")}
    return "wandb" in targets


def setup_wandb(args) -> Optional[Any]:
    if not should_use_wandb(args.report_to):
        logger.info("W&B disabled (report_to=%s).", args.report_to)
        return None

    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"

    return wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args),
        reinit=True,
    )


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


def resolve_dtype(args):
    if args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        logger.info("Using BF16 precision.")
        return torch.bfloat16
    if args.bf16:
        logger.warning("BF16 requested but unsupported; using FP32.")
    logger.info("Using FP32 precision.")
    return torch.float32


def load_predictor(args, dtype):
    tokenizer = AutoTokenizer.from_pretrained(args.predictor_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    order = ["flash_attention_2", "sdpa", "eager"]
    requested = args.attn_implementation
    tried = []
    for impl in order[order.index(requested):]:
        try:
            if impl == "flash_attention_2" and dtype == torch.float32:
                raise ValueError("flash_attention_2 requires BF16/FP16")
            model = AutoModelForSequenceClassification.from_pretrained(
                args.predictor_model,
                trust_remote_code=True,
                attn_implementation=impl,
                torch_dtype=dtype,
            )
            if impl != requested:
                logger.warning("Falling back to attention implementation: %s", impl)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            return tokenizer, model, device, impl
        except Exception as e:
            tried.append((impl, str(e)))
            logger.warning("Failed to load predictor with %s: %s", impl, e)

    detail = "; ".join([f"{impl}: {err}" for impl, err in tried])
    raise RuntimeError(f"Failed to load predictor model. {detail}")


def load_generation_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No generation rows found in {path}")
    return rows


def predict_scores(
    sequences: List[str],
    tokenizer,
    model,
    device: str,
    max_length: int,
    batch_size: int,
) -> List[List[float]]:
    outputs_list: List[List[float]] = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        enc = tokenizer(
            batch,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.inference_mode():
            logits = model(**enc).logits
        outputs_list.extend(logits.detach().float().cpu().tolist())
    return outputs_list


def build_scored_rows(rows: List[Dict[str, Any]], generated_scores, reference_scores):
    scored_rows = []
    for row, gen_score, ref_score in zip(rows, generated_scores, reference_scores):
        delta = [g - r for g, r in zip(gen_score, ref_score)]
        scored_row = dict(row)
        scored_row["generated_prediction"] = gen_score
        scored_row["reference_prediction"] = ref_score
        scored_row["prediction_delta"] = delta
        scored_row["generated_prediction_sum"] = sum(gen_score)
        scored_row["reference_prediction_sum"] = sum(ref_score)
        scored_row["prediction_delta_sum"] = sum(delta)
        if row.get("source_label") is not None:
            true_label = row["source_label"]
            if isinstance(true_label, list) and len(true_label) == len(gen_score):
                scored_row["generated_abs_error"] = [abs(g - t) for g, t in zip(gen_score, true_label)]
                scored_row["reference_abs_error"] = [abs(r - t) for r, t in zip(ref_score, true_label)]
        scored_rows.append(scored_row)
    return scored_rows


def summarise(rows: List[Dict[str, Any]], effective_attn: str) -> Dict[str, Any]:
    dim = len(rows[0]["generated_prediction"])
    summary: Dict[str, Any] = {
        "num_rows": len(rows),
        "attn_implementation": effective_attn,
    }

    for i in range(dim):
        summary[f"mean_generated_prediction_label_{i}"] = sum(row["generated_prediction"][i] for row in rows) / len(rows)
        summary[f"mean_reference_prediction_label_{i}"] = sum(row["reference_prediction"][i] for row in rows) / len(rows)
        summary[f"mean_prediction_delta_label_{i}"] = sum(row["prediction_delta"][i] for row in rows) / len(rows)

    summary["mean_generated_prediction_sum"] = sum(row["generated_prediction_sum"] for row in rows) / len(rows)
    summary["mean_reference_prediction_sum"] = sum(row["reference_prediction_sum"] for row in rows) / len(rows)
    summary["mean_prediction_delta_sum"] = sum(row["prediction_delta_sum"] for row in rows) / len(rows)
    summary["positive_delta_rate"] = sum(1 for row in rows if row["prediction_delta_sum"] > 0) / len(rows)

    top_rows = sorted(rows, key=lambda x: x["generated_prediction_sum"], reverse=True)[:20]
    summary["top_generated_prediction_sum"] = top_rows[0]["generated_prediction_sum"] if top_rows else None
    summary["top_candidates_preview"] = [
        {
            "source_id": row.get("source_id"),
            "condition_token": row.get("condition_token"),
            "activity_bucket": row.get("activity_bucket"),
            "generated_prediction": row["generated_prediction"],
            "reference_prediction": row["reference_prediction"],
            "prediction_delta": row["prediction_delta"],
            "generated_sequence": row.get("generated_sequence"),
        }
        for row in top_rows[:5]
    ]

    group_key = None
    if any(row.get("activity_bucket") is not None for row in rows):
        group_key = "activity_bucket"
    elif any(row.get("condition_token") is not None for row in rows):
        group_key = "condition_token"

    if group_key is not None:
        grouped_summary: Dict[str, Dict[str, Any]] = {}
        group_values = sorted({str(row[group_key]) for row in rows if row.get(group_key) is not None})
        for value in group_values:
            group_rows = [row for row in rows if str(row.get(group_key)) == value]
            grouped_summary[value] = {
                "num_rows": len(group_rows),
                "mean_generated_prediction_sum": sum(row["generated_prediction_sum"] for row in group_rows) / len(group_rows),
                "mean_reference_prediction_sum": sum(row["reference_prediction_sum"] for row in group_rows) / len(group_rows),
                "mean_prediction_delta_sum": sum(row["prediction_delta_sum"] for row in group_rows) / len(group_rows),
                "positive_delta_rate": sum(1 for row in group_rows if row["prediction_delta_sum"] > 0) / len(group_rows),
            }
            for i in range(dim):
                grouped_summary[value][f"mean_generated_prediction_label_{i}"] = (
                    sum(row["generated_prediction"][i] for row in group_rows) / len(group_rows)
                )
                grouped_summary[value][f"mean_reference_prediction_label_{i}"] = (
                    sum(row["reference_prediction"][i] for row in group_rows) / len(group_rows)
                )
                grouped_summary[value][f"mean_prediction_delta_label_{i}"] = (
                    sum(row["prediction_delta"][i] for row in group_rows) / len(group_rows)
                )
        summary[f"by_{group_key}"] = grouped_summary
    return summary


def save_outputs(args, rows: List[Dict[str, Any]], summary: Dict[str, Any]):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    details_path = output_dir / "scoring_details.jsonl"
    with open(details_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_ready(row), ensure_ascii=False) + "\n")

    summary_path = output_dir / "scoring_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(summary), f, indent=2, ensure_ascii=False)

    top_path = output_dir / "top_candidates.json"
    top_rows = sorted(rows, key=lambda x: x["generated_prediction_sum"], reverse=True)[:20]
    with open(top_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(top_rows), f, indent=2, ensure_ascii=False)

    logger.info("Saved scoring details to %s", details_path)
    logger.info("Saved scoring summary to %s", summary_path)
    logger.info("Saved top candidates to %s", top_path)


def main():
    args = parse_args()
    run = setup_wandb(args)
    dtype = resolve_dtype(args)
    tokenizer, model, device, effective_attn = load_predictor(args, dtype)
    rows = load_generation_rows(args.generation_details_path)

    generated_sequences = [row.get("generated_sequence") or (row["prompt"] + row["generated"]) for row in rows]
    reference_sequences = [row.get("reference_sequence") or (row["prompt"] + row["target"]) for row in rows]

    generated_scores = predict_scores(
        generated_sequences, tokenizer, model, device, args.max_length, args.batch_size
    )
    reference_scores = predict_scores(
        reference_sequences, tokenizer, model, device, args.max_length, args.batch_size
    )

    scored_rows = build_scored_rows(rows, generated_scores, reference_scores)
    summary = summarise(scored_rows, effective_attn)
    summary["predictor_model"] = args.predictor_model
    summary["generation_details_path"] = args.generation_details_path
    summary["device"] = device
    summary["bf16"] = args.bf16

    save_outputs(args, scored_rows, summary)
    logger.info("Predictor scoring summary: %s", summary)

    if run is not None:
        wandb.log(summary)
        table = wandb.Table(
            columns=[
                "source_id",
                "generated_prediction",
                "reference_prediction",
                "prediction_delta",
                "generated_prediction_sum",
                "generated_sequence",
            ]
        )
        for row in sorted(scored_rows, key=lambda x: x["generated_prediction_sum"], reverse=True)[:20]:
            table.add_data(
                row.get("source_id"),
                row["generated_prediction"],
                row["reference_prediction"],
                row["prediction_delta"],
                row["generated_prediction_sum"],
                row.get("generated_sequence"),
            )
        wandb.log({"top_scored_candidates": table})
        run.finish()


if __name__ == "__main__":
    main()
