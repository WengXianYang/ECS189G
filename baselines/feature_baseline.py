"""Simple-feature baseline for headline A/B test prediction.

Extracts ~30 interpretable features (length, punctuation, numbers, curiosity-gap
words, sentiment cues, etc.) from each headline, builds (feat_A - feat_B) difference
vectors, and trains regularised logistic regression on the dev set.

No API key needed — runs entirely offline.

Examples (from repo root):
  python baselines/feature_baseline.py
  python baselines/feature_baseline.py --reddit xian_data_prep/inputs/reddit_pairs.csv
  python baselines/feature_baseline.py --out baselines/

Output files (in --out directory, default: baselines/):
  features_dev.csv        pair_id, p_A  (386 rows)  — for --dev-baseline flag
  features_test.csv       pair_id, p_A  (1543 rows) — for --baseline flag
  features_reddit.csv     pair_id, p_A  (500 rows)  — if --reddit is given
  feature_importance.csv  feature, coefficient       — useful for RQ3 / H1
  features_meta.json      run metadata
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# Precompile all regexes once at module level for speed
_RE_DIGIT         = re.compile(r"\d")
_RE_LISTICLE      = re.compile(r"^\d+\s")
_RE_ALL_CAPS      = re.compile(r"\b[A-Z]{2,}\b")
_RE_YOU           = re.compile(r"\b(you|your|you're|yourself)\b", re.I)
_RE_THIS_THESE    = re.compile(r"\b(this|these)\b", re.I)
_RE_SUPERLATIVE   = re.compile(
    r"\b(most|best|worst|greatest|biggest|smallest|fastest|"
    r"amazing|incredible|unbelievable|insane|shocking|stunning|"
    r"mind.?blowing|jaw.?dropping|breathtaking|extraordinary|"
    r"powerful|ultimate|perfect|brilliant|genius)\b", re.I)
_RE_NEGATIVE      = re.compile(
    r"\b(never|no one|nobody|nothing|won't|can't|don't|isn't|"
    r"shouldn't|fail|wrong|bad|worst|terrible|horrible|devastating|"
    r"awful|tragic|disaster|crisis|danger|warning|stop)\b", re.I)
_RE_CURIOSITY     = re.compile(
    r"\b(what happens|you won't believe|here'?s why|the reason|"
    r"find out|discover|revealed|secret|truth about|little.?known|"
    r"surprisingly|actually|really|turns out)\b", re.I)
_RE_POSITIVE      = re.compile(
    r"\b(love|beautiful|amazing|perfect|brilliant|hero|brave|"
    r"inspiring|heartwarming|awesome|fantastic|wonderful|incredible|"
    r"joy|happy|celebrate|win|victory|hope|better|best)\b", re.I)
_RE_IMPERATIVE    = re.compile(
    r"^(watch|listen|look|see|read|meet|check|find|discover|"
    r"take|try|stop|start|remember|imagine|think|consider|learn|"
    r"know|understand|help|join|share|sign|vote)\b", re.I)
_RE_HOW           = re.compile(r"^how\b", re.I)
_RE_WHY           = re.compile(r"^why\b", re.I)
_RE_WHAT          = re.compile(r"^what\b", re.I)
_RE_WHEN          = re.compile(r"^when\b", re.I)
_RE_ELLIPSIS      = re.compile(r"\.{3}|…")
_RE_DASH          = re.compile(r"—|–|-")
_RE_PUNCT         = re.compile(r"[!?.,;:\"'…—–\-]")


def extract_features(headline: str) -> dict[str, float]:
    """Extract a fixed-length feature dict from one headline string."""
    h     = headline.strip()
    lower = h.lower()
    words = h.split()
    n     = max(len(words), 1)

    return {
        # ── Length ──────────────────────────────────────────────────────
        "n_chars":            float(len(h)),
        "n_words":            float(len(words)),
        "avg_word_len":       float(np.mean([len(w) for w in words])) if words else 0.0,
        "max_word_len":       float(max((len(w) for w in words), default=0)),

        # ── Punctuation / form ───────────────────────────────────────────
        "has_question":       float("?" in h),
        "has_exclamation":    float("!" in h),
        "has_ellipsis":       float(bool(_RE_ELLIPSIS.search(h))),
        "has_colon":          float(":" in h),
        "has_dash":           float(bool(_RE_DASH.search(h))),
        "has_quotes":         float('"' in h or "'" in h or "\u2018" in h or "\u201c" in h),
        "n_punctuation":      float(len(_RE_PUNCT.findall(h))),

        # ── Numbers ──────────────────────────────────────────────────────
        "has_number":         float(bool(_RE_DIGIT.search(h))),
        "has_listicle":       float(bool(_RE_LISTICLE.match(h))),   # "10 Things..."

        # ── Capitalisation ───────────────────────────────────────────────
        "n_all_caps_words":   float(len(_RE_ALL_CAPS.findall(h))),
        "title_case_ratio":   float(sum(1 for w in words if w.istitle()) / n),
        "all_lower":          float(h == lower),

        # ── Engagement / persuasion cues ─────────────────────────────────
        "has_you":            float(bool(_RE_YOU.search(h))),
        "has_this_these":     float(bool(_RE_THIS_THESE.search(h))),
        "has_superlative":    float(bool(_RE_SUPERLATIVE.search(h))),
        "has_negative":       float(bool(_RE_NEGATIVE.search(h))),
        "has_curiosity_gap":  float(bool(_RE_CURIOSITY.search(h))),
        "has_positive":       float(bool(_RE_POSITIVE.search(h))),

        # ── Structure / first-word ───────────────────────────────────────
        "starts_with_how":    float(bool(_RE_HOW.match(h))),
        "starts_with_why":    float(bool(_RE_WHY.match(h))),
        "starts_with_what":   float(bool(_RE_WHAT.match(h))),
        "starts_with_when":   float(bool(_RE_WHEN.match(h))),
        "is_imperative":      float(bool(_RE_IMPERATIVE.match(h))),
        "is_question":        float(h.rstrip().endswith("?")),

        # ── Readability proxies ──────────────────────────────────────────
        "long_headline":      float(len(words) >= 12),   # unusually long
        "short_headline":     float(len(words) <= 5),    # unusually short
    }


def pair_diff_features(h_a: str, h_b: str) -> np.ndarray:
    """Feature vector for one pair = feat(A) - feat(B).

    Direction: positive values favour A winning; logistic regression learns which
    direction predicts the winner. Feature names are shared with feat(A).
    """
    fa = extract_features(h_a)
    fb = extract_features(h_b)
    keys = sorted(fa.keys())
    return np.array([fa[k] - fb[k] for k in keys], dtype=np.float32), keys


def build_matrix(df: pd.DataFrame, col_a: str, col_b: str) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix (n_pairs × n_features) and return (X, feature_names)."""
    rows, feature_names = [], None
    for _, row in df.iterrows():
        diff, names = pair_diff_features(row[col_a], row[col_b])
        rows.append(diff)
        if feature_names is None:
            feature_names = names
    return np.stack(rows), feature_names  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 12).tolist(),
            cv=5,
            scoring="accuracy",
            max_iter=2000,
            random_state=42,
            n_jobs=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_upworthy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "test_id" in df.columns, f"Expected Upworthy CSV, got: {list(df.columns)}"
    return df


def load_reddit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "pair_id" in df.columns, f"Expected Reddit CSV, got: {list(df.columns)}"
    return df


def save_predictions(pair_ids, p_A: np.ndarray, path: Path) -> None:
    pd.DataFrame({"pair_id": pair_ids, "p_A": p_A.round(6)}).to_csv(path, index=False)
    print(f"  saved {len(p_A)} predictions → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[feature_baseline] loading pair files...")
    dev_df  = load_upworthy(Path(args.dev))
    test_df = load_upworthy(Path(args.test))
    print(f"  dev  pairs: {len(dev_df)}")
    print(f"  test pairs: {len(test_df)}")

    reddit_df = None
    if args.reddit:
        reddit_df = load_reddit(Path(args.reddit))
        print(f"  reddit pairs: {len(reddit_df)}")

    # ── Build feature matrices ────────────────────────────────────────────
    print("\n  extracting features...")
    X_dev,  feat_names = build_matrix(dev_df,  "headline_A", "headline_B")
    X_test, _          = build_matrix(test_df, "headline_A", "headline_B")
    y_dev = (dev_df["true_winner"] == "A").astype(int).values
    print(f"  feature matrix: {X_dev.shape[1]} features × {X_dev.shape[0]} dev pairs")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n  training (5-fold CV over regularisation C)...")
    pipe = build_pipeline()

    cv_scores = cross_val_score(pipe, X_dev, y_dev, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipe.fit(X_dev, y_dev)
    best_C = float(pipe.named_steps["lr"].C_[0])
    print(f"  best C: {best_C:.5f}")

    in_sample_acc = (pipe.predict(X_dev) == y_dev).mean()
    print(f"  in-sample accuracy (dev): {in_sample_acc:.4f}  "
          f"(overfit; report the CV number above)")

    # ── Feature importance ────────────────────────────────────────────────
    coef = pipe.named_steps["lr"].coef_[0]
    importance = sorted(
        zip(feat_names, coef.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    imp_df = pd.DataFrame(importance, columns=["feature", "coefficient"])
    imp_path = out_dir / "feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)

    print(f"\n  top 10 features by |coefficient| (positive → A more likely to win):")
    for feat, c in importance[:10]:
        bar = "+" * min(int(abs(c) * 20), 30) if c >= 0 else "-" * min(int(abs(c) * 20), 30)
        print(f"    {feat:30s}  {c:+.4f}  {bar}")

    # ── Predict ───────────────────────────────────────────────────────────
    p_A_dev  = pipe.predict_proba(X_dev)[:, 1]
    p_A_test = pipe.predict_proba(X_test)[:, 1]

    save_predictions(dev_df["test_id"],  p_A_dev,  out_dir / "features_dev.csv")
    save_predictions(test_df["test_id"], p_A_test, out_dir / "features_test.csv")

    # ── Reddit cross-domain ───────────────────────────────────────────────
    reddit_acc = None
    if reddit_df is not None:
        X_reddit,  _ = build_matrix(reddit_df, "title_a", "title_b")
        p_A_reddit   = pipe.predict_proba(X_reddit)[:, 1]
        save_predictions(reddit_df["pair_id"], p_A_reddit, out_dir / "features_reddit.csv")
        y_reddit     = (reddit_df["winner"] == "A").astype(int).values
        reddit_acc   = float((pipe.predict(X_reddit) == y_reddit).mean())
        print(f"\n  Reddit cross-domain accuracy (Upworthy-trained → Reddit): {reddit_acc:.4f}")

    # ── Save metadata ─────────────────────────────────────────────────────
    meta = {
        "n_features":       len(feat_names),
        "feature_names":    feat_names,
        "best_C":           best_C,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std":  float(cv_scores.std()),
        "in_sample_acc":    float(in_sample_acc),
        "reddit_accuracy":  reddit_acc,
        "n_dev":            int(len(dev_df)),
        "n_test":           int(len(test_df)),
        "dev_path":         str(args.dev),
        "test_path":        str(args.test),
    }
    meta_path = out_dir / "features_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n  feature importance → {imp_path}")
    print(f"  metadata          → {meta_path}")
    print("\nDone. Usage with metrics.py:")
    print(f"  python tools/metrics.py \\")
    print(f"    --pairs {args.test} \\")
    print(f"    --baseline features=baselines/features_test.csv \\")
    print(f"    --dev-pairs {args.dev} \\")
    print(f"    --dev-baseline features=baselines/features_dev.csv \\")
    print(f"    --out reports/baseline_features")


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
        "--out",
        default="baselines/",
        help="Output directory (default: baselines/)",
    )
    args = ap.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

