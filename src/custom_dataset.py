import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Sequence, Tuple

from pathlib import Path
import pandas as pd


DEFAULT_CONDITION_TOKENS: Tuple[str, ...] = ("<sp0>", "<sp1>", "<sp2>")


def split_condition_prefix(
    text: str,
    condition_tokens: Optional[Sequence[str]] = None,
) -> Tuple[str, str]:
    if not condition_tokens:
        return "", text

    remaining = text
    prefix_parts: List[str] = []
    sorted_tokens = sorted({token for token in condition_tokens if token}, key=len, reverse=True)

    while remaining:
        matched = False
        for token in sorted_tokens:
            if remaining.startswith(token):
                prefix_parts.append(token)
                remaining = remaining[len(token):]
                matched = True
                break
        if not matched:
            break

    return "".join(prefix_parts), remaining


def normalize_sequence_text(
    text: Any,
    uppercase: bool = True,
    strip: bool = True,
    conditioned_input: bool = False,
    condition_tokens: Optional[Sequence[str]] = None,
) -> str:
    value = "" if text is None else str(text)
    if strip:
        value = value.strip()

    if not value:
        return value

    if not conditioned_input:
        return value.upper() if uppercase else value

    prefix, dna_suffix = split_condition_prefix(
        value,
        condition_tokens=condition_tokens or DEFAULT_CONDITION_TOKENS,
    )
    if uppercase:
        dna_suffix = dna_suffix.upper()
    return prefix + dna_suffix


class ParquetSequenceDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        sequence_col: str = "sequence",
        uppercase: bool = True,
        strip: bool = True,
        dropna: bool = True,
        limit: Optional[int] = None,
        conditioned_input: bool = False,
        condition_tokens: Optional[Sequence[str]] = None,
    ):
        if limit is not None and limit <= 0:
            raise ValueError("limit must be a positive integer.")

        p = Path(parquet_path)
        paths: List[Path]
        if p.is_dir():
            paths = sorted(p.glob("*.parquet"))
            if not paths:
                raise FileNotFoundError(f"No parquet files under: {parquet_path}")
        else:
            if not p.exists():
                raise FileNotFoundError(parquet_path)
            paths = [p]

        sequences: List[str] = []
        remaining = limit
        for fp in paths:
            try:
                df = pd.read_parquet(fp, columns=[sequence_col])
            except Exception as e:
                raise ValueError(f"Failed to read column '{sequence_col}' from {fp}: {e}") from e

            if dropna:
                df = df[df[sequence_col].notna()]

            s = df[sequence_col].astype(str)
            s = s.map(
                lambda value: normalize_sequence_text(
                    text=value,
                    uppercase=uppercase,
                    strip=strip,
                    conditioned_input=conditioned_input,
                    condition_tokens=condition_tokens,
                )
            )

            if remaining is not None:
                if remaining <= 0:
                    break
                s = s.iloc[:remaining]
                remaining -= len(s)

            sequences.extend(s.tolist())

            if remaining is not None and remaining <= 0:
                break

        if not sequences:
            raise ValueError(
                "No sequences found after filtering. Check parquet_path and sequence_col."
            )

        self.seqs = sequences

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {"text": self.seqs[idx]}


class SequenceDataCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int = 16384,
        pad_to_multiple_of: Optional[int] = None,
        add_special_tokens: bool = True,  # Let tokenizer add <s> and </s>
        conditioned_input: bool = False,
        condition_tokens: Optional[Sequence[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.add_special_tokens = add_special_tokens
        self.conditioned_input = conditioned_input
        self.condition_tokens = tuple(condition_tokens or DEFAULT_CONDITION_TOKENS)

        # Ensure pad_token is set, even if the tokenizer doesn't register it.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<pad>"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]

        # Right-truncate to a multiple of tokenizer.k before tokenization.
        k = self.tokenizer.k
        if self.conditioned_input:
            processed_texts = []
            for text in texts:
                prefix, dna_suffix = split_condition_prefix(
                    text,
                    condition_tokens=self.condition_tokens,
                )
                usable_length = len(dna_suffix) - (len(dna_suffix) % k)
                processed_texts.append(prefix + dna_suffix[:usable_length])
            texts = processed_texts
        else:
            texts = [t[:len(t) - (len(t) % k)] for t in texts]

        enc = self.tokenizer(
            texts,
            add_special_tokens=self.add_special_tokens,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
