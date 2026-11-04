"""Shared loaders for pair files and model results.

Result rows (one JSON object per line, files named results_{shard}_{model}_{method}.jsonl):
    pair_id, shard, model, method, order, sample_idx, choice, stated_conf,
    raw_output, latency, timestamp
Optional: temperature, runner, dataset.

Conventions (agree with run.py):
- `order` is "AB" (headline A shown first) or "BA" (headline B shown first).
- `choice` is the ORIGINAL label "A" or "B" from the pair file, already mapped back from the
  shown position. null/None means the output could not be parsed.
- `stated_conf` is the model's confidence that its choice is the winner, 0.5-1.0
  (0-1 or 0-100 also accepted; 50-100 is rescaled). null if missing.
- The greedy (temperature 0) call is `temperature == 0`, or `sample_idx == 0` if
  `temperature` is absent. Other sample_idx values are the repeated samples.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["pair_id", "model", "method", "order", "sample_idx"]
REQUIRED = ["pair_id", "shard", "model", "method", "order", "sample_idx", "choice", "stated_conf"]


def load_pairs(path: str | Path) -> pd.DataFrame:
    """Return a standard truth table for an Upworthy or Reddit pair file.

    Columns: pair_id, dataset, winner, value_A, value_B, gap, significant, year, extra cols.
    value_* is CTR for Upworthy and score for Reddit.
    """
    df = pd.read_csv(path)
    if "test_id" in df.columns:  # Upworthy (Yurun / Xian format)
        out = pd.DataFrame({
            "pair_id": df["test_id"].astype(str),
            "dataset": "upworthy",
            "winner": df["true_winner"].astype(str).str.upper(),
            "value_A": df["ctr_A"].astype(float),
            "value_B": df["ctr_B"].astype(float),
            "significant": df["is_significant"].astype(str).str.lower().eq("true")
            if "is_significant" in df else np.nan,
            "year": pd.to_datetime(df["created_at"], format="mixed").dt.year,
        })
    elif "image_id" in df.columns:  # Reddit (Gaeun format)
        out = pd.DataFrame({
            "pair_id": df["pair_id"].astype(str),
            "dataset": "reddit",
            "winner": df["winner"].astype(str).str.upper(),
            "value_A": df["score_a"].astype(float),
            "value_B": df["score_b"].astype(float),
            "significant": np.nan,
            "year": pd.to_datetime(df["time_a"], unit="s").dt.year,
            "first_posted": df["first_posted"],
        })
    else:
        raise ValueError(f"Unknown pair file format: {path}")
    out["gap"] = (out["value_A"] - out["value_B"]).abs()
    if out["pair_id"].duplicated().any():
        raise ValueError(f"Duplicate pair_id in {path}")
    return out


def normalize_conf(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    x = float(x)
    if x > 1.0:
        x = x / 100.0
    return float(np.clip(x, 0.0, 1.0))


def load_results(patterns: list[str]) -> pd.DataFrame:
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise FileNotFoundError(f"No result files match {patterns}")
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    rows.append({"_file": f, "_line": ln, "_bad_json": True})
                    continue
                r["_file"], r["_line"], r["_bad_json"] = f, ln, False
                rows.append(r)
    df = pd.DataFrame(rows)
    for c in REQUIRED + ["temperature", "runner", "dataset"]:
        if c not in df:
            df[c] = np.nan
    df["pair_id"] = df["pair_id"].astype(str)
    df["choice"] = df["choice"].where(df["choice"].isin(["A", "B"]), None)
    df["conf"] = df["stated_conf"].map(normalize_conf)
    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="coerce")
    if df["temperature"].notna().any():
        df["greedy"] = pd.to_numeric(df["temperature"], errors="coerce").eq(0)
    else:
        df["greedy"] = df["sample_idx"].eq(0)
    return df
