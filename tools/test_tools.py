"""Small hand-checked tests. Run: python -m pytest tools/test_tools.py -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import load_results  # noqa: E402
from metrics import apply_policy, choose_threshold, predictions_from_results, score_frame, wilson  # noqa: E402


def _write(tmp_path, rows):
    f = tmp_path / "results_A_m_zs.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(f)


def _row(pid, order, idx, choice, conf):
    return {"pair_id": pid, "shard": "A", "model": "m", "method": "zs", "order": order,
            "sample_idx": idx, "choice": choice, "stated_conf": conf}


def test_wilson_known_value():
    lo, hi = wilson(80, 100)
    assert abs(lo - 0.7112) < 1e-3 and abs(hi - 0.8666) < 1e-3


def test_signals_hand_computed(tmp_path):
    rows = [
        _row("p1", "AB", 0, "A", 0.9), _row("p1", "BA", 0, "A", 0.7),   # greedy: pA 0.9, 0.7 -> stated 0.8
        _row("p1", "AB", 1, "A", 0.6), _row("p1", "BA", 1, "B", 0.6),   # all 4 calls: 3 of 4 pick A
        _row("p2", "AB", 0, "A", 80), _row("p2", "BA", 0, "B", 60),     # 0-100 scale; picks first both times
        _row("p2", "AB", 1, None, None),                                  # parse failure ignored
    ]
    pred = predictions_from_results(load_results([_write(tmp_path, rows)])).set_index("pair_id")
    assert np.isclose(pred.loc["p1", "p_stated"], 0.8)
    assert np.isclose(pred.loc["p1", "p_consistency"], 0.75)
    assert np.isclose(pred.loc["p1", "p_combined"], 0.775)
    assert pred.loc["p1", "order_consistent"] == 1.0
    assert np.isclose(pred.loc["p2", "p_stated"], (0.8 + 0.4) / 2)
    assert pred.loc["p2", "order_consistent"] == 0.0
    assert pred.loc["p2", "picked_first_rate"] == 1.0
    assert pred.loc["p2", "n_valid_calls"] == 2


def test_tie_breaks_to_greedy_ab_and_regret():
    pred = pd.DataFrame({"model": "m", "method": "zs", "pair_id": ["p1", "p2"],
                         "p_combined": [0.5, 0.2], "greedy_AB": ["B", "A"]})
    truth = pd.DataFrame({"pair_id": ["p1", "p2"], "winner": ["B", "A"], "value_A": [0.01, 0.03],
                          "value_B": [0.02, 0.01], "significant": [True, False]})
    d = score_frame(pred, truth, "combined").set_index("pair_id")
    assert d.loc["p1", "pred"] == "B" and d.loc["p1", "correct"] == 1   # tie -> greedy AB choice
    assert d.loc["p2", "pred"] == "B" and d.loc["p2", "correct"] == 0
    assert np.isclose(d.loc["p2", "regret"], 0.02)


def test_policy_threshold():
    d = pd.DataFrame({"conf": [0.95, 0.9, 0.8, 0.6], "correct": [1, 1, 0, 0],
                      "regret": [0, 0, 0.01, 0.02]})
    tau = choose_threshold(d, target_acc=0.66, min_n=1)
    assert tau == 0.8                       # top 3 have accuracy 2/3
    p = apply_policy(d, tau)
    assert p["n_skipped_tests"] == 3 and np.isclose(p["regret_per_pair"], 0.01 / 4)
    assert choose_threshold(d, target_acc=1.01, min_n=1) == np.inf
