"""Build Reddit title-pair dataset from SNAP redditSubmissions.csv.gz.

Usage: python reddit_pairs.py path/to/redditSubmissions.csv.gz [out_dir]

Settings below are placeholders until the team rules meeting confirms them.
"""
import sys, re, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

# ---- settings (confirm at rules meeting) ----
WINNER_METRIC = "score"      # upvotes - downvotes
N_PAIRS = 500
SEED = 42
MIN_TOTAL_VOTES = 1          # drop posts nobody voted on

src = Path(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)
counts = {}

df = pd.read_csv(src, compression="infer", on_bad_lines="skip", low_memory=False)
df.columns = [c.strip().lstrip("#") for c in df.columns]  # raw header is "#image_id"
import gzip
with gzip.open(src, "rt", errors="replace") as fh:
    n_lines = sum(1 for _ in fh) - 1
counts["raw_lines_in_file"] = n_lines
counts["malformed_lines_skipped"] = n_lines - len(df)  # rows with extra commas
counts["raw_rows"] = len(df)
counts["raw_images"] = int(df["image_id"].nunique())

# ---- clean ----
num = ["unixtime", "score", "total_votes", "number_of_upvotes", "number_of_downvotes", "number_of_comments"]
for c in num:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["title"] = df["title"].astype(str).str.strip()
df["subreddit"] = df["subreddit"].astype(str).str.strip().str.lower()
bad_title = df["title"].str.lower().isin(["", "nan", "[deleted]", "[removed]"])
df = df[~bad_title & df["subreddit"].ne("nan")]
df = df.dropna(subset=["image_id", "unixtime", WINNER_METRIC])
counts["after_missing_filter"] = len(df)
df = df.drop_duplicates(subset=["reddit_id"]) if "reddit_id" in df else df
counts["after_dedup"] = len(df)
df = df[df["total_votes"].fillna(0) >= MIN_TOTAL_VOTES]
counts["after_min_votes"] = len(df)
df["title_norm"] = df["title"].str.lower().map(lambda t: re.sub(r"[^a-z0-9]+", "", t))

# ---- candidate pairs: same image + subreddit, different normalized titles ----
rows = []
for (img, sub), g in df.groupby(["image_id", "subreddit"]):
    if g["title_norm"].nunique() < 2:
        continue
    g = g.sample(frac=1, random_state=int(rng.integers(1e9)))
    first = g.iloc[0]
    other = g[g["title_norm"] != first["title_norm"]].iloc[0]
    rows.append((img, sub, first, other))
counts["image_subreddit_groups_with_pair"] = len(rows)

# one pair per image
cand = pd.DataFrame([{"image_id": r[0], "i": i} for i, r in enumerate(rows)])
pick = cand.groupby("image_id")["i"].apply(lambda s: int(rng.choice(s.values)))
rows = [rows[i] for i in pick.values]
counts["images_with_pair"] = len(rows)

recs = []
for img, sub, a, b in rows:
    if a[WINNER_METRIC] == b[WINNER_METRIC]:
        continue  # ties dropped
    # random A/B assignment so position carries no information
    if rng.random() < 0.5:
        a, b = b, a
    recs.append({
        "pair_id": f"rd_{img}_{sub}",
        "image_id": img, "subreddit": sub,
        "title_a": a["title"], "title_b": b["title"],
        "reddit_id_a": a.get("reddit_id"), "reddit_id_b": b.get("reddit_id"),
        f"{WINNER_METRIC}_a": a[WINNER_METRIC], f"{WINNER_METRIC}_b": b[WINNER_METRIC],
        "votes_a": a["total_votes"], "votes_b": b["total_votes"],
        "comments_a": a["number_of_comments"], "comments_b": b["number_of_comments"],
        "time_a": int(a["unixtime"]), "time_b": int(b["unixtime"]),
        "first_posted": "A" if a["unixtime"] < b["unixtime"] else "B",
        "gap_hours": abs(a["unixtime"] - b["unixtime"]) / 3600,
        "abs_diff": abs(a[WINNER_METRIC] - b[WINNER_METRIC]),
        "winner": "A" if a[WINNER_METRIC] > b[WINNER_METRIC] else "B",
    })
pairs = pd.DataFrame(recs)
counts["pairs_after_tie_drop"] = len(pairs)

pairs.to_csv(out / "reddit_pairs_all.csv", index=False)
sample = pairs.sample(n=min(N_PAIRS, len(pairs)), random_state=SEED).reset_index(drop=True)
sample.to_csv(out / "reddit_pairs.csv", index=False)
counts["final_pairs"] = len(sample)
counts["final_first_posted_wins"] = float((sample["winner"] == sample["first_posted"]).mean())
counts["final_winner_A_share"] = float((sample["winner"] == "A").mean())
counts["final_subreddits"] = int(sample["subreddit"].nunique())
counts["sha256_reddit_pairs"] = hashlib.sha256((out / "reddit_pairs.csv").read_bytes()).hexdigest()
counts["settings"] = {"winner_metric": WINNER_METRIC, "n_pairs": N_PAIRS, "seed": SEED, "min_total_votes": MIN_TOTAL_VOTES}
(out / "reddit_counts.json").write_text(json.dumps(counts, indent=2, default=str))
print(json.dumps(counts, indent=2, default=str))
