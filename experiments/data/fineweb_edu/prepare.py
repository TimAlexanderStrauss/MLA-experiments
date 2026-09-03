"""
Download and tokenize the FineWeb-Edu dataset for the MLA ablation experiment.

Output:
  train.bin  — ~500M tokens (uint16, GPT-2 BPE)
  val.bin    — ~  5M tokens (uint16, GPT-2 BPE)

The two splits are disjoint: after tokenization the token stream is cut at a
fixed index (first 500M tokens -> train, next 5M -> val), so the split is
deterministic and val.bin never contaminates train.bin across any run.

Usage (run once on the desktop PC before training):
  cd experiments
  python data/fineweb_edu/prepare.py

Requirements: datasets, tiktoken, tqdm
Estimated disk space: ~1 GB for 505M uint16 tokens
Estimated time:       15–30 min depending on download speed
"""

import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET      = "HuggingFaceFW/fineweb-edu"
DATASET_NAME = "sample-10BT"           # ~10B token subset — more than enough
TRAIN_TOKENS = 500_000_000             # 500M training tokens
VAL_TOKENS   =   5_000_000            #   5M validation tokens
TOTAL_TOKENS = TRAIN_TOKENS + VAL_TOKENS
SHARD_SIZE   = 100_000_000            # Process in 100M-token shards to manage RAM

OUT_DIR = Path(__file__).parent        # data/fineweb_edu/
TRAIN_BIN = OUT_DIR / "train.bin"
VAL_BIN   = OUT_DIR / "val.bin"


def tokenize(doc: dict, enc: tiktoken.Encoding) -> np.ndarray:
    """Tokenize a single document; prepend <|endoftext|> as document separator."""
    tokens = [enc.eot_token]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    arr = np.array(tokens, dtype=np.uint16)
    # GPT-2 vocab is 50257, safely fits in uint16 (max 65535)
    assert arr.max() < 2**16
    return arr


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_BIN.exists() and VAL_BIN.exists():
        train_len = len(np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r"))
        val_len   = len(np.memmap(VAL_BIN,   dtype=np.uint16, mode="r"))
        print(f"Found existing splits:  train={train_len:,}  val={val_len:,}")
        if train_len >= TRAIN_TOKENS and val_len >= VAL_TOKENS:
            print("Splits look complete. Delete .bin files to re-download.")
            return

    print(f"Loading {DATASET} ({DATASET_NAME}) …")
    ds = load_dataset(DATASET, name=DATASET_NAME, split="train", streaming=True)

    enc = tiktoken.get_encoding("gpt2")

    # Collect tokens into a single buffer up to TOTAL_TOKENS
    all_tokens: list[np.ndarray] = []
    total_collected = 0

    for doc in tqdm(ds, desc="Tokenizing", unit=" docs"):
        arr = tokenize(doc, enc)
        all_tokens.append(arr)
        total_collected += len(arr)
        if total_collected >= TOTAL_TOKENS:
            break

    print(f"Tokenized {total_collected:,} tokens total.")

    combined = np.concatenate(all_tokens)[:TOTAL_TOKENS]
    train_arr = combined[:TRAIN_TOKENS]
    val_arr   = combined[TRAIN_TOKENS:TRAIN_TOKENS + VAL_TOKENS]

    print(f"Writing {TRAIN_BIN.name}  ({len(train_arr):,} tokens) …")
    train_arr.tofile(TRAIN_BIN)

    print(f"Writing {VAL_BIN.name}    ({len(val_arr):,} tokens) …")
    val_arr.tofile(VAL_BIN)

    print("Done.")
    print(f"  train.bin : {TRAIN_BIN.stat().st_size / 1e9:.2f} GB")
    print(f"  val.bin   : {VAL_BIN.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
