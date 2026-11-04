"""Accuracy, calibration, risk-coverage and skipping-policy metrics.

Example:
  python tools/metrics.py --pairs data/splits/upworthy_test.csv --results "results/test/*.jsonl" \
      --dev-pairs data/splits/upworthy_dev.csv --dev-results "results/dev/*.jsonl" \
      --baseline features=baselines/features_test.csv --out reports/test

Each model x method is scored with three confidence signals:
  stated       mean of the greedy calls' stated confidence over both orders
  consistency  share of all valid samples (both orders) that pick A
  combined     average of the two
Baseline files: CSV with columns pair_id, p_A (probability A wins). Scored as signal "prob".

Skipping policy: if confidence >= tau, ship the predicted winner without running the test;
otherwise run the test and ship the true winner. tau is chosen on dev as the lowest threshold
whose selective accuracy on dev reaches --target-acc. CTR regret on a shipped pair is
max(value) - value(predicted); a pair that is tested has regret 0 and costs one test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from common import load_pairs, load_results
except ImportError:  # imported as tools.metrics
    from tools.common import load_pairs, load_results

SIGNALS = ["stated", "consistency", "combined"]


# ---------------------------------------------------------------- predictions
def predictions_from_results(res: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, method, pair_id) with p_A for each signal and order diagnostics."""
    ok = res[res["choice"].notna()].copy()
    ok["is_A"] = ok["choice"].eq("A").astype(float)
    g = ok[ok["greedy"]].copy()
    g["pA_call"] = np.where(g["is_A"] == 1, g["conf"], 1 - g["conf"])
    g["picked_first"] = (g["choice"] == g["order"].str[0]).astype(float)
    keys = ["model", "method", "pair_id"]

    stated = g.groupby(keys)["pA_call"].mean().rename("p_stated")
    cons = ok.groupby(keys)["is_A"].mean().rename("p_consistency")
    n_valid = ok.groupby(keys).size().rename("n_valid_calls")
    greedy_ab = g[g["order"] == "AB"].groupby(keys)["choice"].first().rename("greedy_AB")
    greedy_ba = g[g["order"] == "BA"].groupby(keys)["choice"].first().rename("greedy_BA")
    first = g.groupby(keys)["picked_first"].mean().rename("picked_first_rate")
    out = pd.concat([stated, cons, n_valid, greedy_ab, greedy_ba, first], axis=1).reset_index()
    out["order_consistent"] = np.where(
        out["greedy_AB"].notna() & out["greedy_BA"].notna(),
        (out["greedy_AB"] == out["greedy_BA"]).astype(float), np.nan)
    out["p_combined"] = out[["p_stated", "p_consistency"]].mean(axis=1, skipna=True)
    return out


def baseline_predictions(name: str, path: str) -> pd.DataFrame:
    b = pd.read_csv(path)
    if not {"pair_id", "p_A"} <= set(b.columns):
        raise ValueError(f"Baseline {path} needs columns pair_id, p_A")
    return pd.DataFrame({"model": name, "method": "baseline", "pair_id": b["pair_id"].astype(str),
                         "p_prob": b["p_A"].astype(float)})


def score_frame(pred: pd.DataFrame, truth: pd.DataFrame, signal: str) -> pd.DataFrame:
    """Join one signal with truth; tie (p == 0.5) breaks to the greedy AB choice, else A."""
    col = f"p_{signal}"
    d = pred[["model", "method", "pair_id", col] + [c for c in ["greedy_AB"] if c in pred]].merge(
        truth, on="pair_id", how="inner")
    d = d[d[col].notna()].copy()
    d["p_A"] = d[col]
    tie = np.isclose(d["p_A"], 0.5)
    pred_A = d["p_A"] > 0.5
    if "greedy_AB" in d:
        pred_A = np.where(tie, d["greedy_AB"].eq("A"), pred_A)
    d["pred"] = np.where(pred_A, "A", "B")
    d["conf"] = np.maximum(d["p_A"], 1 - d["p_A"])
    d["correct"] = (d["pred"] == d["winner"]).astype(int)
    d["y"] = (d["winner"] == "A").astype(int)
    best = np.maximum(d["value_A"], d["value_B"])
    picked = np.where(d["pred"] == "A", d["value_A"], d["value_B"])
    d["regret"] = best - picked
    d["signal"] = signal
    return d


# ---------------------------------------------------------------- metrics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def calibration_bins(d: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(d["conf"], edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(m.sum()),
                     "mean_conf": float(d["conf"][m].mean()) if m.any() else np.nan,
                     "accuracy": float(d["correct"][m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def risk_coverage(d: pd.DataFrame) -> pd.DataFrame:
    """Sort by confidence (desc); coverage k/n, risk = error rate among the k most confident."""
    s = d.sort_values("conf", ascending=False, kind="mergesort")
    err = 1 - s["correct"].to_numpy()
    k = np.arange(1, len(s) + 1)
    return pd.DataFrame({"coverage": k / len(s), "risk": np.cumsum(err) / k,
                         "threshold": s["conf"].to_numpy(),
                         "mean_regret_shipped": np.cumsum(s["regret"].to_numpy()) / k})


def aurc(rc: pd.DataFrame) -> float:
    return float(rc["risk"].mean()) if len(rc) else np.nan


def safe_auroc(conf, correct) -> float:
    return float(roc_auc_score(correct, conf)) if len(set(correct)) == 2 else np.nan


def summarize(d: pd.DataFrame) -> dict:
    n, k = len(d), int(d["correct"].sum())
    lo, hi = wilson(k, n)
    sig = d[d["significant"] == True]  # noqa: E712
    return {
        "n": n, "accuracy": k / n if n else np.nan, "acc_ci_lo": lo, "acc_ci_hi": hi,
        "acc_significant": sig["correct"].mean() if len(sig) else np.nan, "n_significant": len(sig),
        "brier": float(np.mean((d["p_A"] - d["y"]) ** 2)) if n else np.nan,
        "ece": ece(d["conf"].to_numpy(), d["correct"].to_numpy()) if n else np.nan,
        "auroc": safe_auroc(d["conf"], d["correct"]) if n else np.nan,
        "aurc": aurc(risk_coverage(d)) if n else np.nan,
        "mean_conf": d["conf"].mean() if n else np.nan,
        "mean_regret_all_shipped": d["regret"].mean() if n else np.nan,
    }


# ---------------------------------------------------------------- skipping policy
def choose_threshold(dev: pd.DataFrame, target_acc: float, min_n: int = 20) -> float:
    """Lowest tau whose dev selective accuracy >= target with at least min_n pairs covered."""
    rc = risk_coverage(dev)
    rc["n"] = np.arange(1, len(rc) + 1)
    ok = rc[(1 - rc["risk"] >= target_acc) & (rc["n"] >= min_n)]
    if ok.empty:
        return np.inf  # never skip
    return float(ok.iloc[-1]["threshold"])


def apply_policy(d: pd.DataFrame, tau: float) -> dict:
    shipped = d["conf"] >= tau
    n = len(d)
    return {
        "tau": tau, "coverage": shipped.mean() if n else np.nan, "n_skipped_tests": int(shipped.sum()),
        "selective_acc": d.loc[shipped, "correct"].mean() if shipped.any() else np.nan,
        "regret_per_skipped": d.loc[shipped, "regret"].mean() if shipped.any() else np.nan,
        "regret_per_pair": d["regret"].where(shipped, 0.0).mean() if n else np.nan,
        "tests_saved_share": shipped.mean() if n else np.nan,
    }


# ---------------------------------------------------------------- main
def build_scored(pairs_path, result_globs, baselines) -> pd.DataFrame:
    truth = load_pairs(pairs_path)
    frames = []
    if result_globs:
        pred = predictions_from_results(load_results(result_globs))
        for s in SIGNALS:
            frames.append(score_frame(pred, truth, s))
    for name, path in baselines:
        frames.append(score_frame(baseline_predictions(name, path), truth, "prob"))
    if not frames:
        raise SystemExit("Nothing to score: give --results and/or --baseline")
    return pd.concat(frames, ignore_index=True)


def order_diagnostics(result_globs) -> pd.DataFrame:
    if not result_globs:
        return pd.DataFrame()
    pred = predictions_from_results(load_results(result_globs))
    return pred.groupby(["model", "method"]).agg(
        pairs=("pair_id", "nunique"),
        order_consistency=("order_consistent", "mean"),
        picked_first_rate=("picked_first_rate", "mean")).reset_index()


def plot(scored: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for kind in ("reliability", "risk_coverage"):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for (m, me, s), d in scored.groupby(["model", "method", "signal"]):
            label = f"{m}/{me}/{s}"
            if kind == "reliability":
                cb = calibration_bins(d).dropna()
                ax.plot(cb["mean_conf"], cb["accuracy"], marker="o", label=label)
            else:
                rc = risk_coverage(d)
                ax.plot(rc["coverage"], rc["risk"], label=label)
        if kind == "reliability":
            ax.plot([0.5, 1], [0.5, 1], "k--", lw=1)
            ax.set(xlabel="Confidence", ylabel="Accuracy", title="Reliability")
        else:
            ax.set(xlabel="Coverage", ylabel="Risk (error rate)", title="Risk-coverage")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / f"{kind}.png", dpi=150)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--dev-pairs")
    ap.add_argument("--dev-results", nargs="*", default=[])
    ap.add_argument("--baseline", nargs="*", default=[], help="name=path.csv (test)")
    ap.add_argument("--dev-baseline", nargs="*", default=[], help="name=path.csv (dev)")
    ap.add_argument("--target-acc", type=float, default=0.80)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "risk_coverage.csv").unlink(missing_ok=True)
    bl = [tuple(x.split("=", 1)) for x in a.baseline]
    dbl = [tuple(x.split("=", 1)) for x in a.dev_baseline]

    scored = build_scored(a.pairs, a.results, bl)
    dev_scored = build_scored(a.dev_pairs, a.dev_results, dbl) if a.dev_pairs else None

    rows, policy_rows = [], []
    for (m, me, s), d in scored.groupby(["model", "method", "signal"]):
        rows.append({"model": m, "method": me, "signal": s, **summarize(d)})
        if dev_scored is not None:
            dd = dev_scored[(dev_scored.model == m) & (dev_scored.method == me) & (dev_scored.signal == s)]
            if len(dd):
                tau = choose_threshold(dd, a.target_acc)
                policy_rows.append({"model": m, "method": me, "signal": s,
                                    "target_acc": a.target_acc, **apply_policy(d, tau)})
        rc = risk_coverage(d)
        rc.insert(0, "signal", s); rc.insert(0, "method", me); rc.insert(0, "model", m)
        rc.to_csv(out / "risk_coverage.csv", mode="a", index=False,
                  header=not (out / "risk_coverage.csv").exists())
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    policy = pd.DataFrame(policy_rows)
    if len(policy):
        policy.to_csv(out / "policy.csv", index=False)
    diag = order_diagnostics(a.results)
    if len(diag):
        diag.to_csv(out / "order_diagnostics.csv", index=False)
    scored.to_csv(out / "scored_pairs.csv", index=False)
    plot(scored, out)

    with open(out / "report.md", "w", encoding="utf-8") as fh:
        fh.write(f"# Metrics report\n\nPairs: `{a.pairs}`\n\n## Summary\n\n")
        fh.write(summary.round(4).to_markdown(index=False))
        if len(diag):
            fh.write("\n\n## Position bias and order consistency (greedy calls)\n\n")
            fh.write(diag.round(4).to_markdown(index=False))
        if len(policy):
            fh.write(f"\n\n## Skipping policy (tau chosen on dev, target accuracy {a.target_acc})\n\n")
            fh.write(policy.round(5).to_markdown(index=False))
        fh.write("\n\nFigures: reliability.png, risk_coverage.png\n")
    print(json.dumps({"out": str(out), "rows": len(summary)}, indent=1))
    return summary


if __name__ == "__main__":
    main()
