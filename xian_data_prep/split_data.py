"""Create or verify the headline-grouped, stratified Upworthy split (stdlib only)."""

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

BASE = Path(__file__).resolve().parent
SEED = 42
DEV_FRACTION = Fraction(1, 5)
ALGORITHM = "headline-components-stratified-v1"


def normalize(text):
    """The exact ASCII normalization specified in Yurun's rules_spec.md."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or not rows:
        raise ValueError(f"Empty CSV: {path}")
    return fields, rows


def significant(row):
    value = row["is_significant"].lower()
    if value not in {"true", "false"}:
        raise ValueError("is_significant must be True or False")
    return value == "true"


def stratum(row):
    return row["true_winner"] + ("_significant" if significant(row) else "_not_significant")


def headline_groups(rows):
    """Connected components: sharing either headline joins entire pairs transitively."""
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        for column in ("headline_A", "headline_B"):
            headline = normalize(row[column])
            if not headline:
                raise ValueError(f"Empty normalized headline in {row['test_id']}")
            if headline in seen:
                parent[find(index)] = find(seen[headline])
            seen[headline] = index

    components = defaultdict(list)
    for index, row in enumerate(rows):
        components[find(index)].append(row)
    groups = {}
    for component in components.values():
        headlines = sorted({normalize(r[c]) for r in component for c in ("headline_A", "headline_B")})
        group_id = "uwg_" + sha256(json_bytes(headlines))
        groups[group_id] = sorted(component, key=lambda r: r["test_id"])
    return groups


def rank(group_id, seed):
    # SHA ranking avoids dependency on RNG implementation, row order, and Python hash seed.
    return sha256(f"{seed}:{group_id}".encode("utf-8"))


def split_rows(rows, seed=SEED):
    rows = sorted(rows, key=lambda r: r["test_id"])
    if len({r["test_id"] for r in rows}) != len(rows):
        raise ValueError("Upworthy requires exactly one row per test_id")
    for row in rows:
        if row["true_winner"] not in {"A", "B"}:
            raise ValueError("Unknown winner")
        for side in ("A", "B"):
            n, c = int(row[f"impressions_{side}"]), int(row[f"clicks_{side}"])
            if not 0 <= c <= n or n < 500:
                raise ValueError("Invalid Upworthy clicks/impressions")
        difference = int(row["clicks_A"]) * int(row["impressions_B"]) - int(row["clicks_B"]) * int(row["impressions_A"])
        if not difference or row["true_winner"] != ("A" if difference > 0 else "B"):
            raise ValueError("Upworthy winner disagrees with exact CTR")
        if "2013-06-25" <= row["created_at"][:10] <= "2014-01-10":
            raise ValueError("Unreliable-randomization window is present")

    groups = headline_groups(rows)
    totals = Counter(stratum(r) for r in rows)
    n_dev = (len(rows) + 2) // 5  # nearest integer to 20%; no binary float rounding
    targets = {s: n_dev * n // len(rows) for s, n in totals.items()}
    remainder_order = sorted(totals, key=lambda s: (-(n_dev * totals[s] % len(rows)), s))
    for s in remainder_order[:n_dev - sum(targets.values())]:
        targets[s] += 1

    remaining = dict(targets)
    dev_groups = set()
    singletons = defaultdict(list)
    for group_id in sorted(groups, key=lambda g: rank(g, seed)):
        component = groups[group_id]
        if len(component) == 1:
            singletons[stratum(component[0])].append(group_id)
            continue
        contribution = Counter(stratum(r) for r in component)
        chosen = int(rank(group_id, seed), 16) * DEV_FRACTION.denominator < (1 << 256) * DEV_FRACTION.numerator
        if chosen and all(contribution[s] <= remaining[s] for s in contribution):
            dev_groups.add(group_id)
            for s, n in contribution.items():
                remaining[s] -= n
    for s, needed in remaining.items():
        if needed > len(singletons[s]):
            raise ValueError("Cannot meet stratum quotas with these components; use a new reviewed split version")
        dev_groups.update(singletons[s][:needed])

    splits = {"dev": [], "test": []}
    assignments = []
    for group_id, component in sorted(groups.items()):
        name = "dev" if group_id in dev_groups else "test"
        assignments.append({"headline_group_id": group_id, "split": name, "pair_count": len(component)})
        splits[name].extend(dict(row, headline_group_id=group_id) for row in component)
    for name in splits:
        splits[name].sort(key=lambda r: r["test_id"])
    if Counter(stratum(r) for r in splits["dev"]) != Counter(targets):
        raise ValueError("Stratification quotas were not met")
    if len(splits["test"]) < 1500:
        raise ValueError("The rules require at least 1,500 Upworthy test pairs")
    return splits, assignments, targets


def leakage_report(splits, reddit):
    headlines = {
        name: {normalize(r[c]) for r in rows for c in ("headline_A", "headline_B")}
        for name, rows in splits.items()
    }
    test_ids = {name: {r["test_id"] for r in rows} for name, rows in splits.items()}
    group_ids = {name: {r["headline_group_id"] for r in rows} for name, rows in splits.items()}
    reddit_titles = {normalize(r[c]) for r in reddit for c in ("title_a", "title_b")}
    report = {
        "normalized_headline_overlap_dev_test": len(headlines["dev"] & headlines["test"]),
        "test_id_overlap_dev_test": len(test_ids["dev"] & test_ids["test"]),
        "headline_group_overlap_dev_test": len(group_ids["dev"] & group_ids["test"]),
        "reddit_title_overlap_with_upworthy_dev": len(reddit_titles & headlines["dev"]),
        "reddit_title_overlap_with_upworthy_test": len(reddit_titles & headlines["test"]),
        "reddit_usage": "All 500 supplied pairs retained as a separate evaluation domain; no Reddit dev/test split.",
    }
    if any(report[k] for k in ("normalized_headline_overlap_dev_test", "test_id_overlap_dev_test", "headline_group_overlap_dev_test")):
        raise ValueError("Leakage detected")
    return report


def prepare_outputs(upworthy, reddit, seed=SEED):
    fields, rows = read_csv(upworthy)
    _, reddit_rows = read_csv(reddit)
    splits, assignments, targets = split_rows(rows, seed)
    summary = {
        "algorithm": ALGORITHM, "seed": seed, "dev_fraction": "1/5", "test_fraction": "4/5",
        "normalization": "re.sub(r'[^a-z0-9]', '', text.lower())",
        "stratification": "true_winner x is_significant; largest-remainder integer dev quotas",
        "total_pairs": len(rows), "headline_groups": len(assignments),
        "group_size_counts": dict(sorted(Counter(a["pair_count"] for a in assignments).items())),
        "dev_stratum_targets": targets,
        "splits": {
            name: {"pairs": len(part), "winner_counts": dict(Counter(r["true_winner"] for r in part)),
                   "significant_pairs": sum(significant(r) for r in part),
                   "stratum_counts": dict(Counter(stratum(r) for r in part)),
                   "headline_groups": len({r["headline_group_id"] for r in part})}
            for name, part in splits.items()
        },
        "reddit_supplied_pairs": len(reddit_rows),
        "reddit_pairs_within_30_days": sum(abs(int(r["time_a"]) - int(r["time_b"])) <= 30 * 86400 for r in reddit_rows),
    }
    outputs = {
        f"upworthy_{name}.csv": csv_bytes(part, fields + ["headline_group_id"])
        for name, part in splits.items()
    }
    outputs["headline_groups.csv"] = csv_bytes(assignments, ["headline_group_id", "split", "pair_count"])
    outputs["split_summary.json"] = json_bytes(summary)
    outputs["leakage_report.json"] = json_bytes(leakage_report(splits, reddit_rows))
    lock = {
        "schema_version": 1, "algorithm": ALGORITHM, "seed": seed, "dev_fraction": "1/5",
        "scope": "Upworthy dev/test only; supplied Reddit sample retained with documented rule mismatch",
        "inputs_sha256": {"upworthy_pairs.csv": sha256(Path(upworthy).read_bytes()),
                          "reddit_pairs.csv": sha256(Path(reddit).read_bytes())},
        "outputs_sha256": {name: sha256(data) for name, data in sorted(outputs.items())},
        "test_pairs": summary["splits"]["test"]["pairs"],
        "team_confirmation": "Pending independent teammate reproduction; see TEST_CONFIRMATIONS.md",
    }
    return outputs, lock


def write_or_verify(outputs, lock, out_dir, lock_path):
    # Validate everything before writing anything. A mismatch never refreshes the lock.
    lock_data = json_bytes(lock)
    if lock_path.exists() and lock_path.read_bytes() != lock_data:
        raise ValueError("LOCK MISMATCH: inputs, settings, or generated outputs changed. No files written.")
    for name, data in outputs.items():
        target = out_dir / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f"LOCKED FILE MISMATCH: {target}. No files written.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        target = out_dir / name
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(data)
    if not lock_path.exists():
        with lock_path.open("xb") as stream:
            stream.write(lock_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upworthy", type=Path, default=BASE / "inputs/upworthy_pairs.csv")
    parser.add_argument("--reddit", type=Path, default=BASE / "inputs/reddit_pairs.csv")
    parser.add_argument("--out-dir", type=Path, default=BASE / "outputs")
    args = parser.parse_args()
    outputs, lock = prepare_outputs(args.upworthy, args.reddit)
    write_or_verify(outputs, lock, args.out_dir.resolve(), BASE / "TEST_LOCK.json")
    summary = json.loads(outputs["split_summary.json"])
    print(f"Verified: {summary['splits']['dev']['pairs']} dev / {lock['test_pairs']} test pairs; zero normalized headline overlap.")
    print("Upworthy test SHA-256: " + lock["outputs_sha256"]["upworthy_test.csv"])
    print("Team reproduction confirmations remain pending.")


if __name__ == "__main__":
    main()
