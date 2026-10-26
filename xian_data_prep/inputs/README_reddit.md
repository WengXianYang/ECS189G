# Reddit pairs (data prep, owner: Gaeun)

## Source
SNAP Reddit submissions (Lakkaraju, McAuley & Leskovec, ICWSM 2013), file `redditSubmissions.csv.gz`
from https://metromaps.stanford.edu/data/web-Reddit.html. One row per submission; rows with the same
`image_id` are the same image posted again under a different title. Covers July 2008 to January 2013.

Note: the raw header names the first column `#image_id`; the script strips the `#`.

## How to rebuild
```
python reddit_pairs.py redditSubmissions.csv.gz out/
```
Output is deterministic (seed 42). `reddit_counts.json` stores the counts and the SHA-256 of `reddit_pairs.csv`.

## Rules used (placeholders until the rules meeting confirms them)
| Setting | Value |
|---|---|
| Winner metric | `score` (upvotes minus downvotes) |
| Ties | dropped |
| Min total votes per post | 1 |
| Pair definition | same `image_id` and same subreddit, different titles |
| "Different title" | titles differ after lowercasing and removing all non-alphanumeric characters (catches "Bella's first hand job." vs "Bella's First Handjob") |
| Pairs per image | 1, chosen at random with seed 42 |
| A/B order | random, so position carries no information |
| Final sample | 500 pairs, seed 42 |

## Cleaning steps and counts
| Step | Rows / pairs |
|---|---|
| Lines in raw file | 132,308 |
| Malformed lines skipped (extra commas) | 5 |
| Rows read | 132,303 (16,736 images) |
| After dropping missing / deleted / empty titles | 132,302 |
| After dropping duplicate `reddit_id` | 132,209 |
| After min-votes filter | 132,206 |
| Image + subreddit groups with ≥2 distinct titles | 23,990 |
| Images with a pair (1 per image) | 15,744 |
| After dropping score ties | 15,411 (`reddit_pairs_all.csv`) |
| Final sample | 500 (`reddit_pairs.csv`) |

## Columns (`reddit_pairs.csv`)
`pair_id, image_id, subreddit, title_a, title_b, reddit_id_a, reddit_id_b, score_a, score_b,
votes_a, votes_b, comments_a, comments_b, time_a, time_b, first_posted, gap_hours, abs_diff, winner`

`first_posted` is which title was posted first; `gap_hours` is the time between the two posts;
`abs_diff` is |score_a − score_b|.

## Checks done
- All 500 pairs re-checked against the raw file (same image, same subreddit, scores and winner match): 0 failures.
- No duplicate images; no pair with identical titles.
- 20 pairs read by hand. This found near-duplicate titles slipping through, which led to the stricter title rule above.

## Caveats for analysis
- **Posting order is a strong confound.** The earlier post wins 68% of pairs. Report results with
  `first_posted` as a control, or also on a subset balanced by order.
- **Many pairs are low-signal.** 17% have a score gap of 5 or less, and 21% have both scores at 10 or below.
  Consider a minimum `abs_diff` (decide at the rules meeting), and use `abs_diff` for the effect-size comparison in H2.
- **Subreddits are skewed.** About half the pairs are from r/funny, then r/pics, r/gifs, r/wtf, r/aww.
- **Some titles are crude or NSFW.** Some models may refuse or hedge on them; log refusals as a separate outcome rather than dropping them silently.

## Still to do
- Leakage check on the Upworthy dev/test split (after Xian finishes the split on 10/28).
