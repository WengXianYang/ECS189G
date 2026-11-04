"""Protocol isolation, confidence validity and runner integration checks; no model calls."""

import csv
import io
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import protocol, select_examples
from qwen import run_qwen


@pytest.fixture
def cfg():
    return protocol.load_config(verify=False)


def test_json_config_is_valid_yaml(cfg):
    import yaml
    assert yaml.safe_load((ROOT / "config.yaml").read_text())["model"] == cfg["model"]


def test_fixed_examples_are_balanced_dev_rows_and_avoid_all_test_content(cfg):
    _, dev = select_examples.read_csv(ROOT / "xian_data_prep/outputs/upworthy_dev.csv")
    by_id = {r["test_id"]: r for r in dev}
    assert Counter(e["answer"] for e in cfg["_examples"]) == {1: 2, 2: 2}
    assert len({e["content_group_id"] for e in cfg["_examples"]}) == 4
    blocked = set()
    refs = []
    for name in ("xian_data_prep/outputs/upworthy_test.csv", "xian_data_prep/archive/headline-v1/outputs/upworthy_test.csv"):
        _, rows = select_examples.read_csv(ROOT / name)
        refs.extend(rows)
        for row in rows:
            blocked.update(select_examples.content_keys(row))
    chosen = []
    for example in cfg["_examples"]:
        row = by_id[example["pair_id"]]
        chosen.append(row)
        assert example["headline_1"] == row["headline_A"]
        assert example["headline_2"] == row["headline_B"]
        assert example["answer"] == (1 if row["true_winner"] == "A" else 2)
        assert not (select_examples.content_keys(row) & blocked)
        assert "confidence" not in example
    assert select_examples.near_links(chosen, refs) == []


def test_excerpt_exclusion_is_transitive_and_ignores_empty_excerpts():
    rows = [
        {"test_id": "1", "headline_A": "a", "headline_B": "b", "eyecatcher_id": "i1", "excerpt": "Shared!"},
        {"test_id": "2", "headline_A": "c", "headline_B": "d", "eyecatcher_id": "i2", "excerpt": "shared"},
        {"test_id": "3", "headline_A": "e", "headline_B": "f", "eyecatcher_id": "i2", "excerpt": ""},
        {"test_id": "4", "headline_A": "g", "headline_B": "h", "eyecatcher_id": "i4", "excerpt": ""},
    ]
    groups = select_examples.exclusion_components(rows)
    assert groups["1"] == groups["2"] == groups["3"]
    assert groups["4"] != groups["1"]
    assert groups == select_examples.exclusion_components(list(reversed(rows)))


def test_dev_pool_and_shards_exclude_every_demo_related_and_near_test_row(cfg):
    _, source = select_examples.read_csv(ROOT / "xian_data_prep/outputs/upworthy_dev.csv")
    _, pool = select_examples.read_csv(ROOT / "data/prompt_dev.csv")
    _, shards = select_examples.read_csv(ROOT / "data/shards/prompt_dev_shards.csv")
    expected = [r for r in source if r["test_id"] not in cfg["_dev_excluded_ids"]]
    assert pool == expected
    ids = {r["test_id"] for r in pool}
    assert ids == {r["pair_id"] for r in shards}
    assert not (ids & cfg["_test_ids"])
    assert len(pool) == 379


@pytest.mark.parametrize("method", ["zero_shot", "few_shot", "reasoning"])
def test_templates_swap_only_target_and_never_render_outcome_fields(cfg, method):
    pair = {"pair_id": "target", "dataset": "upworthy", "A": 'Alpha {headline_2} "quoted"',
            "B": "Beta\nnew line", "context": "EXCERPT_CANARY", "true_winner": "LABEL_CANARY",
            "clicks_A": "COUNT_CANARY", "shard": "D1"}
    for order, first, second in (("AB", pair["A"], pair["B"]), ("BA", pair["B"], pair["A"])):
        messages = run_qwen.build_messages(cfg, method, pair, order)
        content = messages[-1]["content"]
        target = json.loads(content.split("Current comparison, encoded as JSON data:\n")[1].split("\n\n")[0])
        assert target == {"headline_1": first, "headline_2": second}
        assert all(canary not in content for canary in ("EXCERPT_CANARY", "LABEL_CANARY", "COUNT_CANARY"))
        assert cfg["_confidence"] in content
        assert content.count('"observed_winner"') == (4 if method == "few_shot" else 0)


def test_reddit_uses_explicitly_cross_domain_examples_without_reddit_labels(cfg):
    pair = {"dataset": "reddit", "A": "First", "B": "Second", "context": "r/pics"}
    text = protocol.build_messages(cfg, "few_shot", pair, "AB")[-1]["content"]
    assert "r/pics" in text and "cross-domain" in text
    assert text.count('"observed_winner"') == 4


@pytest.mark.parametrize("value, expected", [(50, .5), (80, .8), (100, 1), (.75, .75),
                                               (49, None), (101, None), (-1, None),
                                               (float("nan"), None), (float("inf"), None), (True, None)])
def test_confidence_is_validated_without_clipping(value, expected):
    assert run_qwen._conf(value) == expected


def test_parser_preserves_choices_with_invalid_confidence_and_ignores_thinking():
    assert run_qwen.parse('<think>{"choice":2,"confidence":99}</think>{"choice":1,"confidence":80}') == (1, .8)
    assert run_qwen.parse('{"choice":2,"confidence":101}') == (2, None)
    assert run_qwen.parse('{"choice":true,"confidence":80}') == (None, None)
    assert run_qwen.parse('choice: 1 confidence: 80') == (None, None)


def test_config_drift_and_resume_contamination_are_rejected(cfg, tmp_path):
    for name in protocol.PROTOCOL_FILES:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    (tmp_path / protocol.LOCK_NAME).write_bytes(protocol.json_bytes(protocol.expected_lock(tmp_path)))
    protocol.verify_lock(tmp_path)
    with (tmp_path / "prompts/confidence.txt").open("a") as stream:
        stream.write("drift")
    with pytest.raises(ValueError, match="lock mismatch"):
        protocol.verify_lock(tmp_path)
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"pair_id": "p", "order": "AB", "sample_idx": 0, "protocol_sha256": "old"}) + "\n")
    with pytest.raises(ValueError, match="different protocol"):
        run_qwen.done_keys(path, {"protocol_sha256": "new"})


def test_sampling_overrides_cannot_change_the_frozen_experiment(cfg):
    with pytest.raises(ValueError, match="frozen"):
        protocol.validate_overrides(SimpleNamespace(sample_temperature=.9), cfg)
    with pytest.raises(ValueError, match="per-method"):
        protocol.validate_overrides(SimpleNamespace(samples=5, methods=["reasoning"]), cfg)


def test_runner_payloads_mapping_sampling_and_resume(cfg, tmp_path, monkeypatch):
    requests = []
    def urlopen(request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        if url.endswith("/api/tags"):
            response = {"models": [{"name": cfg["model"], "digest": cfg["model_digest"]}]}
        elif url.endswith("/api/version"):
            response = {"version": "fixture-server"}
        else:
            body = json.loads(request.data)
            requests.append(body)
            response = {"message": {"content": '{"choice":1,"confidence":80}', "thinking": ""}}
        return io.BytesIO(json.dumps(response).encode())
    monkeypatch.setattr(run_qwen.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(run_qwen, "load_config", lambda path: cfg)
    pairs = tmp_path / "pairs.csv"
    pairs.write_text('test_id,headline_A,headline_B\ntarget,First target,Second target\n')
    args = SimpleNamespace(config=ROOT / "config.yaml", model=None, seed=None, sample_temperature=None,
                           samples=None, methods=["zero_shot", "reasoning"], host="http://fixture",
                           smoke=False, dry_run=False, pairs=str(pairs), shards=None, shard="D1", limit=0,
                           out_dir=str(tmp_path / "results"), workers=1, timeout=1, runner="fixture",
                           keep_thinking=False, progress_every=100)
    assert run_qwen.run(args) == 0
    assert len(requests) == 20  # 2 orders x (1+5) zero-shot; 2 x (1+3) reasoning
    assert sum(r["think"] for r in requests) == 8
    assert all(r["format"]["required"] == ["choice", "confidence"] for r in requests)
    rows = [json.loads(line) for path in (tmp_path / "results").glob("*.jsonl") for line in path.read_text().splitlines()]
    for row in rows:
        assert row["choice"] == row["order"][0]
        assert row["seed"] == 42 + row["sample_idx"]
        assert row["temperature"] == (0 if row["sample_idx"] == 0 else .7)
        assert row["stated_conf"] == .8
        assert row["model_digest"] == cfg["model_digest"]
    assert run_qwen.run(args) == 0
    assert len(requests) == 20


def test_wrong_model_digest_fails_before_inference(cfg, monkeypatch):
    monkeypatch.setattr(protocol.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps({
        "models": [{"name": cfg["model"], "digest": "wrong"}]}).encode()))
    with pytest.raises(ValueError, match="digest"):
        protocol.verify_model("http://fixture", cfg)
