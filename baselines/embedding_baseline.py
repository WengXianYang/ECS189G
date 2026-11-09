"""Embedding baseline for headline A/B test prediction.

Trains logistic regression on (emb_A - emb_B) difference vectors from the dev set,
then predicts P(A wins) for every dev and test pair. Outputs CSVs in the format
expected by tools/metrics.py --baseline.

Supports three embedding providers:
  openai   text-embedding-3-small   (default; ~$0.01 for all pairs)
  google   text-embedding-004       (free tier generous)
  local    TF-IDF bag-of-words      (no API key; useful for quick testing)

Examples (from repo root):
  # full run with OpenAI embeddings (recommended)
  python baselines/embedding_baseline.py --provider openai

  # Google embeddings
  python baselines/embedding_baseline.py --provider google

  # offline/free dry-run with TF-IDF (no API key needed)
  python baselines/embedding_baseline.py --provider local

  # include Reddit cross-domain predictions
  python baselines/embedding_baseline.py --provider openai --reddit xian_data_prep/inputs/reddit_pairs.csv

Output files (all in --out directory, default: baselines/):
  embedding_dev.csv      pair_id, p_A  (386 rows)  — for --dev-baseline flag
  embedding_test.csv     pair_id, p_A  (1543 rows) — for --baseline flag
  embedding_reddit.csv   pair_id, p_A  (500 rows)  — if --reddit is given
  embedding_meta.json    run metadata (provider, model, CV accuracy, etc.)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------

def embed_openai(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed texts via OpenAI API. Batches of 100, polite rate-limit delay."""
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY env var is not set.")
    client = OpenAI(api_key=api_key)
    all_embs: list[list[float]] = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        resp  = client.embeddings.create(model=model, input=batch)
        # API returns embeddings in the same order as input
        all_embs.extend([e.embedding for e in resp.data])
        done = min(i + 100, len(texts))
        print(f"  embedded {done}/{len(texts)}", flush=True)
        if done < len(texts):
            time.sleep(0.3)  # stay well inside rate limits
    return np.array(all_embs, dtype=np.float32)


def embed_google(texts: list[str], model: str = "models/text-embedding-004") -> np.ndarray:
    """Embed texts via Google Generative AI API. Batches of 100."""
    import google.generativeai as genai
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise SystemExit("ERROR: GOOGLE_API_KEY env var is not set.")
    genai.configure(api_key=api_key)
    all_embs: list[list[float]] = []
    for i in range(0, len(texts), 100):
        batch  = texts[i : i + 100]
        result = genai.embed_content(model=model, content=batch)
        all_embs.extend(result["embedding"])
        done = min(i + 100, len(texts))
        print(f"  embedded {done}/{len(texts)}", flush=True)
        if done < len(texts):
            time.sleep(0.2)
    return np.array(all_embs, dtype=np.float32)


def embed_local(texts: list[str], **_) -> np.ndarray:
    """TF-IDF bag-of-words embeddings — no API key, good for offline testing.
    Lower accuracy expected (~52-55%) but enough to test the pipeline.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("  using local TF-IDF (no API key required)", flush=True)
    vec = TfidfVectorizer(max_features=4096, sublinear_tf=True, ngram_range=(1, 2))
    mat = vec.fit_transform(texts).toarray()
    return mat.astype(np.float32)


PROVIDERS = {
    "openai": (embed_openai, "text-embedding-3-small"),
    "google": (embed_google, "models/text-embedding-004"),
    "local":  (embed_local,  "tfidf-4096"),
}


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_feature_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Pair feature = emb_A - emb_B (difference vector).

    Direction: positive dimensions favour A winning, negative favour B.
    Difference-only keeps dimensionality at D (not 3D), reducing overfitting
    risk on the 386-pair dev set.
    """
    return emb_a - emb_b


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """Regularised logistic regression with 5-fold CV over C.

    StandardScaler is critical for embedding differences whose scales vary by dim.
    LogisticRegressionCV searches 10 C values log-spaced in [0.001, 10],
    picking the one that maximises 5-fold accuracy on the dev set.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 10).tolist(),
            cv=5,
            scoring="accuracy",
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_upworthy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "test_id" in df.columns, f"Expected Upworthy CSV, got columns: {list(df.columns)}"
    return df


def load_reddit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "pair_id" in df.columns, f"Expected Reddit CSV, got columns: {list(df.columns)}"
    return df


def save_predictions(pair_ids: pd.Series | list, p_A: np.ndarray, path: Path) -> None:
    out = pd.DataFrame({"pair_id": pair_ids, "p_A": p_A.round(6)})
    out.to_csv(path, index=False)
    print(f"  saved {len(out)} predictions → {path}")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    provider_name = args.provider
    if provider_name not in PROVIDERS:
        raise SystemExit(f"Unknown provider {provider_name!r}. Choose: {list(PROVIDERS)}")
    embed_fn, default_model = PROVIDERS[provider_name]
    model_name = args.model or default_model

    # ── Load pair files ────────────────────────────────────────────────────
    print(f"\n[embedding_baseline] provider={provider_name}  model={model_name}")
    dev_df  = load_upworthy(Path(args.dev))
    test_df = load_upworthy(Path(args.test))
    print(f"  dev  pairs: {len(dev_df)}")
    print(f"  test pairs: {len(test_df)}")

    reddit_df = None
    if args.reddit:
        reddit_df = load_reddit(Path(args.reddit))
        print(f"  reddit pairs: {len(reddit_df)}")

    # ── Collect all headlines to embed in one API call ─────────────────────
    # Order: [dev_A...], [dev_B...], [test_A...], [test_B...], [reddit_a...], [reddit_b...]
    n_dev  = len(dev_df)
    n_test = len(test_df)

    texts: list[str] = (
        list(dev_df["headline_A"])
        + list(dev_df["headline_B"])
        + list(test_df["headline_A"])
        + list(test_df["headline_B"])
    )
    if reddit_df is not None:
        texts += list(reddit_df["title_a"]) + list(reddit_df["title_b"])

    # Deduplicate to save API calls: embed unique texts, then re-index
    unique_texts  = list(dict.fromkeys(texts))   # preserves order, removes dups
    text_to_idx   = {t: i for i, t in enumerate(unique_texts)}

    print(f"\n  embedding {len(unique_texts)} unique texts "
          f"({len(texts)} total, {len(texts)-len(unique_texts)} duplicates saved)...")
    t0         = time.time()
    all_embs   = embed_fn(unique_texts, model_name) if provider_name != "local" \
                 else embed_fn(unique_texts)
    print(f"  embedding done in {time.time()-t0:.1f}s  shape={all_embs.shape}")

    def get_embs(text_list: list[str]) -> np.ndarray:
        idxs = [text_to_idx[t] for t in text_list]
        return all_embs[idxs]

    # ── Build feature matrices ─────────────────────────────────────────────
    emb_dev_a  = get_embs(list(dev_df["headline_A"]))
    emb_dev_b  = get_embs(list(dev_df["headline_B"]))
    emb_test_a = get_embs(list(test_df["headline_A"]))
    emb_test_b = get_embs(list(test_df["headline_B"]))

    X_dev  = build_feature_matrix(emb_dev_a, emb_dev_b)
    X_test = build_feature_matrix(emb_test_a, emb_test_b)
    y_dev  = (dev_df["true_winner"] == "A").astype(int).values

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n  training logistic regression (5-fold CV over regularisation C)...")
    pipe = build_pipeline()

    cv_scores = cross_val_score(pipe, X_dev, y_dev, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipe.fit(X_dev, y_dev)
    best_C = float(pipe.named_steps["lr"].C_[0])
    print(f"  best C (regularisation): {best_C:.5f}")

    in_sample_acc = (pipe.predict(X_dev) == y_dev).mean()
    print(f"  in-sample accuracy (dev): {in_sample_acc:.4f}  "
          f"(note: overestimates; use CV above for reporting)")

    # ── Predict ───────────────────────────────────────────────────────────
    p_A_dev  = pipe.predict_proba(X_dev)[:, 1]   # class-1 = A wins
    p_A_test = pipe.predict_proba(X_test)[:, 1]

    save_predictions(dev_df["test_id"],  p_A_dev,  out_dir / "embedding_dev.csv")
    save_predictions(test_df["test_id"], p_A_test, out_dir / "embedding_test.csv")

    # ── Reddit cross-domain ───────────────────────────────────────────────
    reddit_cv_acc = None
    if reddit_df is not None:
        emb_rd_a  = get_embs(list(reddit_df["title_a"]))
        emb_rd_b  = get_embs(list(reddit_df["title_b"]))
        X_reddit  = build_feature_matrix(emb_rd_a, emb_rd_b)
        p_A_reddit = pipe.predict_proba(X_reddit)[:, 1]
        save_predictions(reddit_df["pair_id"], p_A_reddit, out_dir / "embedding_reddit.csv")

        # Evaluate cross-domain (train on Upworthy dev, test on Reddit)
        y_reddit = (reddit_df["winner"] == "A").astype(int).values
        rd_acc   = (pipe.predict(X_reddit) == y_reddit).mean()
        print(f"\n  Reddit cross-domain accuracy (Upworthy-trained → Reddit): {rd_acc:.4f}")
        reddit_cv_acc = float(rd_acc)

    # ── Save metadata ─────────────────────────────────────────────────────
    meta = {
        "provider":         provider_name,
        "model":            model_name,
        "feature":          "emb_A - emb_B",
        "embedding_dim":    int(all_embs.shape[1]),
        "n_dev":            int(n_dev),
        "n_test":           int(n_test),
        "n_unique_texts":   len(unique_texts),
        "best_C":           best_C,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std":  float(cv_scores.std()),
        "in_sample_acc":    float(in_sample_acc),
        "reddit_accuracy":  reddit_cv_acc,
        "dev_path":         str(args.dev),
        "test_path":        str(args.test),
    }
    meta_path = out_dir / "embedding_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n  metadata saved → {meta_path}")
    print("\nDone. Usage with metrics.py:")
    print(f"  python tools/metrics.py \\")
    print(f"    --pairs {args.test} \\")
    print(f"    --baseline embedding=baselines/embedding_test.csv \\")
    print(f"    --dev-pairs {args.dev} \\")
    print(f"    --dev-baseline embedding=baselines/embedding_dev.csv \\")
    print(f"    --out reports/baseline_embedding")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dev",
        default="xian_data_prep/outputs/upworthy_dev.csv",
        help="Dev split CSV (default: xian_data_prep/outputs/upworthy_dev.csv)",
    )
    ap.add_argument(
        "--test",
        default="xian_data_prep/outputs/upworthy_test.csv",
        help="Test split CSV (default: xian_data_prep/outputs/upworthy_test.csv)",
    )
    ap.add_argument(
        "--reddit",
        default=None,
        help="Reddit pairs CSV for cross-domain predictions (optional)",
    )
    ap.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "google", "local"],
        help="Embedding provider (default: openai)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Embedding model override (default: provider's default model)",
    )
    ap.add_argument(
        "--out",
        default="baselines/",
        help="Output directory for CSV and metadata files (default: baselines/)",
    )
    args = ap.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

