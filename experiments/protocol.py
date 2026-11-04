"""Load, render, verify and freeze the shared headline-prediction protocol."""

import argparse
import csv
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_NAME = "EXPERIMENT_LOCK.json"
PROTOCOL_FILES = [
    "config.yaml", "few_shot.json", "experiments/protocol.py", "experiments/select_examples.py",
    "experiments/few_shot_selection.json", "experiments/dev_prompt_review.json",
    "data/prompt_dev.csv", "data/shards/prompt_dev_shards.csv",
    "data/shards/upworthy_dev_shards.csv", "xian_data_prep/TEST_LOCK.json",
    "xian_data_prep/outputs/upworthy_dev.csv", "xian_data_prep/outputs/upworthy_test.csv",
    "xian_data_prep/archive/headline-v1/outputs/upworthy_test.csv",
    "qwen/run_qwen.py", "qwen/qwen_config.json",
    "prompts/system.txt", "prompts/confidence.txt", "prompts/upworthy.txt", "prompts/reddit.txt",
    "prompts/zero_shot.txt", "prompts/few_shot.txt", "prompts/reasoning.txt",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def expected_lock(root=ROOT):
    cfg = json.loads((root / "config.yaml").read_bytes())
    return {
        "schema_version": 1,
        "protocol_id": cfg["protocol_id"],
        "frozen_on": cfg["frozen_on"],
        "model": cfg["model"], "model_digest": cfg["model_digest"],
        "live_dev_pilot": "pending; offline checks do not establish predictive performance",
        "sha256": {name: digest((root / name).read_bytes()) for name in sorted(PROTOCOL_FILES)},
    }


def verify_lock(root=ROOT):
    path = root / LOCK_NAME
    recorded = json.loads(path.read_bytes())
    expected = expected_lock(root)
    if recorded != expected:
        changed = [name for name, value in expected["sha256"].items() if recorded.get("sha256", {}).get(name) != value]
        raise ValueError("Protocol lock mismatch: " + ", ".join(changed or ["metadata"]))
    return digest(path.read_bytes())


def load_config(path=None, verify=True):
    path = Path(path or ROOT / "config.yaml").resolve()
    # config.yaml deliberately uses JSON syntax, a YAML subset, for stdlib-only runners.
    cfg = json.loads(path.read_bytes())
    if "protocol_config" in cfg:
        path = (path.parent / cfg["protocol_config"]).resolve()
        cfg = json.loads(path.read_bytes())
    root = path.parent
    if path.name != "config.yaml" or cfg.get("schema_version") != 1:
        raise ValueError("Use the shared config.yaml or qwen/qwen_config.json pointer")
    cfg["_protocol_sha256"] = verify_lock(root) if verify else "offline-unfrozen"
    cfg["_system"] = (root / cfg["system_file"]).read_text(encoding="utf-8").strip()
    cfg["_confidence"] = (root / cfg["confidence_file"]).read_text(encoding="utf-8").strip()
    cfg["_tasks"] = {name: (root / part["task_file"]).read_text(encoding="utf-8").strip()
                     for name, part in cfg["datasets"].items()}
    cfg["_templates"] = {name: (root / part["prompt_file"]).read_text(encoding="utf-8").strip()
                         for name, part in cfg["methods"].items()}
    examples = json.loads((root / cfg["few_shot_file"]).read_bytes())
    cfg["_examples"] = examples["examples"]
    report = json.loads((root / "experiments/few_shot_selection.json").read_bytes())
    cfg["_dev_excluded_ids"] = set(report["excluded_from_dev_scoring"])
    cfg["_test_ids"] = set()
    with (root / cfg["datasets"]["upworthy"]["test_pairs"]).open(encoding="utf-8", newline="") as stream:
        cfg["_test_ids"] = {r["test_id"] for r in csv.DictReader(stream)}
    return cfg


def build_messages(cfg, method, pair, order):
    if order not in cfg["sampling"]["orders"]:
        raise ValueError(f"Unknown order: {order}")
    dataset = pair["dataset"]
    if dataset not in cfg["datasets"] or method not in cfg["methods"]:
        raise ValueError("Unknown dataset or method")
    first, second = (pair["A"], pair["B"]) if order == "AB" else (pair["B"], pair["A"])
    task = cfg["_tasks"][dataset].format(context=pair.get("context", ""))
    example_blocks = []
    if cfg["methods"][method].get("few_shot"):
        if len(cfg["_examples"]) != 4:
            raise ValueError("The frozen few-shot method requires four examples")
        for index, example in enumerate(cfg["_examples"], 1):
            example_blocks.append(f"Example {index} (Upworthy headline A/B test):\n" + json.dumps({
                "headline_1": example["headline_1"], "headline_2": example["headline_2"],
                "observed_winner": example["answer"]}, ensure_ascii=False))
        if dataset == "reddit":
            example_blocks.insert(0, "These examples come from Upworthy click-through tests. They are cross-domain demonstrations, not Reddit score observations.")
    user = cfg["_templates"][method].format(
        task=task, examples="\n\n".join(example_blocks),
        headlines_json=json.dumps({"headline_1": first, "headline_2": second}, ensure_ascii=False),
        confidence_question=cfg["_confidence"],
    )
    return [{"role": "system", "content": cfg["_system"]}, {"role": "user", "content": user}]


def validate_overrides(args, cfg):
    for flag, expected in (("model", cfg["model"]), ("seed", cfg["sampling"]["seed"]),
                           ("sample_temperature", cfg["sampling"]["sample_temperature"])):
        if getattr(args, flag, None) is not None and getattr(args, flag) != expected:
            raise ValueError(f"--{flag.replace('_', '-')} differs from frozen config.yaml")
    if getattr(args, "samples", None) is not None:
        if any(cfg["methods"][method]["samples"] != args.samples for method in args.methods):
            raise ValueError("--samples differs from the frozen per-method counts")


def verify_model(host, cfg, timeout=10):
    with urllib.request.urlopen(host + "/api/tags", timeout=timeout) as response:
        tags = json.load(response)
    matches = [m for m in tags.get("models", []) if m.get("name") == cfg["model"] or m.get("model") == cfg["model"]]
    expected = cfg["model_digest"].removeprefix("sha256:")
    if not matches or any(m.get("digest", "").removeprefix("sha256:") != expected for m in matches):
        raise ValueError("Installed model digest does not match config.yaml; verify the exact Qwen artifact before running")
    with urllib.request.urlopen(host + "/api/version", timeout=timeout) as response:
        version = json.load(response).get("version")
    return {"model_digest": cfg["model_digest"], "ollama_version": version}


def review_dev(root=ROOT):
    cfg = load_config(root / "config.yaml", verify=False)
    with (root / cfg["datasets"]["upworthy"]["dev_pairs"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    counts = {method: 0 for method in cfg["methods"]}
    lengths = {method: 0 for method in cfg["methods"]}
    for row in rows:
        pair = {"pair_id": row["test_id"], "A": row["headline_A"], "B": row["headline_B"],
                "context": "", "dataset": "upworthy"}
        assert pair["pair_id"] not in cfg["_dev_excluded_ids"] | cfg["_test_ids"]
        for method in cfg["methods"]:
            for order in cfg["sampling"]["orders"]:
                messages = build_messages(cfg, method, pair, order)
                assert len(messages) == 2 and cfg["_confidence"] in messages[1]["content"]
                text = "\n".join(m["content"] for m in messages)
                assert "{headlines_json}" not in text and "{confidence_question}" not in text
                counts[method] += 1
                lengths[method] = max(lengths[method], len(text))
    return {
        "dev_pool_sha256": digest((root / cfg["datasets"]["upworthy"]["dev_pairs"]).read_bytes()),
        "dev_pairs_checked": len(rows), "rendered_targets_by_method": counts,
        "maximum_prompt_characters_by_method": lengths,
        "example_count": len(cfg["_examples"]),
        "confidence_scale": "50-100 percent probability of selected option matching the observed winner",
        "manual_design_choices": [
            "Same target and confidence definition for every method; native thinking enabled only for reasoning.",
            "Headline JSON data separates source text from instructions; target outcomes and excerpts are not rendered.",
            "Few-shot examples contain observed winners without invented confidence labels.",
            "Demonstration-related dev groups and near-test duplicates are excluded from every method's dev scoring pool.",
        ],
        "live_dev_model_pilot": {"status": "pending", "reason": "No local Ollama service or configured API model access during preparation."},
        "performance_claims": "None: offline prompt validation does not measure accuracy or calibration.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="create the lock after offline checks; refuses to overwrite drift")
    parser.add_argument("--review-dev", action="store_true", help="write the offline dev input review")
    args = parser.parse_args()
    if args.review_dev:
        target = ROOT / "experiments/dev_prompt_review.json"
        data = json_bytes(review_dev())
        if target.exists() and target.read_bytes() != data:
            raise ValueError("Dev review changed; use a new reviewed protocol version")
        target.write_bytes(data)
    if args.freeze:
        data = json_bytes(expected_lock())
        target = ROOT / LOCK_NAME
        if target.exists() and target.read_bytes() != data:
            raise ValueError("Frozen protocol changed; use a new reviewed protocol version")
        target.write_bytes(data)
    if (ROOT / LOCK_NAME).exists():
        print("Verified protocol SHA-256: " + verify_lock())
    elif not args.review_dev:
        raise ValueError("No protocol lock; create it with --review-dev --freeze")


if __name__ == "__main__":
    main()
