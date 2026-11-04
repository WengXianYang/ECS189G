"""Select fixed dev demonstrations and an evaluation pool without using test outcomes."""

import argparse
import csv
import difflib
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SIMILARITY = 0.90


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def content_keys(row):
    keys = {("headline", normalize(row[c])) for c in ("headline_A", "headline_B")}
    keys.add(("image", row["eyecatcher_id"].strip()))
    excerpt = normalize(row["excerpt"])
    if excerpt:
        keys.add(("excerpt", excerpt))
    return keys


def exclusion_components(rows):
    """Use stricter excerpt links for demonstration exclusion, including transitivity."""
    parent = {row["test_id"]: row["test_id"] for row in rows}

    def find(pid):
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]
            pid = parent[pid]
        return pid

    seen = {}
    for row in sorted(rows, key=lambda r: r["test_id"]):
        pid = row["test_id"]
        for key in sorted(content_keys(row)):
            if key in seen:
                a, b = sorted((find(pid), find(seen[key])))
                parent[b] = a
            seen[key] = pid
    return {pid: find(pid) for pid in parent}


def near_links(candidates, references, threshold=SIMILARITY):
    """Compare normalized headlines only; neither labels nor outcome statistics are read."""
    reference_heads = defaultdict(set)
    for row in references:
        for column in ("headline_A", "headline_B"):
            reference_heads[normalize(row[column])].add(row["test_id"])
    matchers = [(text, sorted(ids), difflib.SequenceMatcher(None, "", text, autojunk=False))
                for text, ids in sorted(reference_heads.items())]
    links = []
    for row in sorted(candidates, key=lambda r: r["test_id"]):
        found = None
        for column in ("headline_A", "headline_B"):
            text = normalize(row[column])
            for other, ids, matcher in matchers:
                if not text or not other or 2 * min(len(text), len(other)) / (len(text) + len(other)) < threshold:
                    continue
                matcher.set_seq1(text)
                if matcher.quick_ratio() < threshold:
                    continue
                score = matcher.ratio()
                if score < threshold:
                    score = max(score, difflib.SequenceMatcher(None, other, text, autojunk=False).ratio())
                if score >= threshold:
                    found = {"dev_id": row["test_id"], "reference_id": ids[0], "similarity": round(score, 6)}
                    break
            if found:
                break
        if found:
            links.append(found)
    return links


def stratum(row):
    return row["true_winner"] + ("_significant" if row["is_significant"].lower() == "true" else "_not_significant")


def prepare(root=ROOT):
    base = root / "xian_data_prep"
    paths = {
        "source": base / "inputs/upworthy_pairs.csv",
        "dev": base / "outputs/upworthy_dev.csv",
        "test": base / "outputs/upworthy_test.csv",
        "v1_test": base / "archive/headline-v1/outputs/upworthy_test.csv",
        "shards": root / "data/shards/upworthy_dev_shards.csv",
    }
    hashes = {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths.values()}
    _, rows = read_csv(paths["source"])
    fields, dev = read_csv(paths["dev"])
    # Project test rows to content and IDs immediately: outcomes never enter selection.
    content_columns = ("test_id", "headline_A", "headline_B", "eyecatcher_id", "excerpt")
    reference = {}
    for name in ("test", "v1_test"):
        _, raw = read_csv(paths[name])
        reference[name] = [{key: row[key] for key in content_columns} for row in raw]
    components = exclusion_components(rows)
    blocked = {components[r["test_id"]] for part in reference.values() for r in part}
    exact_safe = [r for r in dev if components[r["test_id"]] not in blocked]
    near = near_links(exact_safe, reference["test"] + reference["v1_test"])
    near_ids = {r["dev_id"] for r in near}
    candidates = [r for r in exact_safe if r["test_id"] not in near_ids
                  and normalize(r["headline_A"]) != normalize(r["headline_B"])
                  and all(15 <= len(r[c].strip()) <= 250 for c in ("headline_A", "headline_B"))]
    rank = lambda r: hashlib.sha256(f"{SEED}:{r['test_id']}".encode()).hexdigest()
    chosen = []
    used_components = set()
    for target in ("A_not_significant", "B_not_significant", "A_significant", "B_significant"):
        eligible = sorted((r for r in candidates if stratum(r) == target
                           and components[r["test_id"]] not in used_components), key=rank)
        if not eligible:
            raise ValueError(f"No safe example for {target}")
        chosen.append(eligible[0])
        used_components.add(components[eligible[0]["test_id"]])
    demo_related = {r["test_id"] for r in dev if components[r["test_id"]] in used_components}
    dev_near = near_links(dev, reference["test"])
    excluded = demo_related | {r["dev_id"] for r in dev_near}
    pool = [r for r in dev if r["test_id"] not in excluded]
    pool_ids = {r["test_id"] for r in pool}
    shard_fields, shard_rows = read_csv(paths["shards"])
    examples = {
        "schema_version": 1,
        "source": "xian_data_prep/outputs/upworthy_dev.csv",
        "source_sha256": hashes["xian_data_prep/outputs/upworthy_dev.csv"],
        "seed": SEED,
        "examples": [{"pair_id": r["test_id"], "dataset": "upworthy",
                      "content_group_id": r["content_group_id"],
                      "headline_1": r["headline_A"], "headline_2": r["headline_B"],
                      "answer": 1 if r["true_winner"] == "A" else 2,
                      "stratum": stratum(r)} for r in chosen],
    }
    report = {
        "source_sha256": hashes,
        "reported_flagged_test_count": 51,
        "reported_flagged_ids_available": False,
        "exclusion_policy": "Conservative superset: every v1 and v2 test pair, linked transitively by normalized headline, image ID, or nonempty normalized excerpt; also direct near-duplicate headlines.",
        "near_duplicate_rule": "max of both SequenceMatcher directions on ASCII-normalized headlines, autojunk=False, ratio >= 0.90",
        "test_outcomes_used": False,
        "dev_pairs": len(dev),
        "exact_safe_candidates": len(exact_safe),
        "eligible_candidate_strata": dict(Counter(stratum(r) for r in candidates)),
        "candidate_near_test_links": near,
        "selected_ids": [r["test_id"] for r in chosen],
        "selected_winner_counts": dict(Counter(r["true_winner"] for r in chosen)),
        "demo_related_dev_ids": sorted(demo_related),
        "dev_near_current_test_links": dev_near,
        "excluded_from_dev_scoring": sorted(excluded),
        "dev_scoring_pairs": len(pool),
        "selection": "One example per winner/significance stratum, distinct exclusion components; SHA-256 rank of seed:test_id; no model-performance search.",
    }
    return {
        "few_shot.json": json_bytes(examples),
        "experiments/few_shot_selection.json": json_bytes(report),
        "data/prompt_dev.csv": csv_bytes(pool, fields),
        "data/shards/prompt_dev_shards.csv": csv_bytes([r for r in shard_rows if r["pair_id"] in pool_ids], shard_fields),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="create missing artifacts; never overwrite different bytes")
    args = parser.parse_args()
    outputs = prepare()
    for name, data in outputs.items():
        path = ROOT / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Selection drift: {name}; use a reviewed new protocol version")
        if not path.exists() and not args.write:
            raise ValueError(f"Missing {name}; generate with --write")
    if args.write:
        for name, data in outputs.items():
            path = ROOT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
    report = json.loads(outputs["experiments/few_shot_selection.json"])
    print(f"Verified four fixed dev examples; {report['dev_scoring_pairs']} independent dev scoring pairs")


if __name__ == "__main__":
    main()
