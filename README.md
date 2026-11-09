# ECS 189G: Can LLMs Predict A/B Test Results, and Do They Know When They're Wrong?

Team: Gaeun Yu, Yurun Shi, Xian Yang Weng

We evaluate off-the-shelf LLMs on predicting the winners of real headline A/B tests (Upworthy Research Archive),
with Reddit same-image resubmissions as a second, observational domain. Focus: accuracy, calibration, the cost of
skipping tests when the model is confident, and in which settings predictions work.

## Repository layout
```
upworthy/   Upworthy pair builder, rules, 1,929 pairs   (Yurun)
reddit/     Reddit pair builder, 500 pairs, counts       (Gaeun)
xian_data_prep/   Headline/image-grouped Upworthy split v2, test lock, and audit (Xian)
```

## Data
| Dataset | Source | Raw file (not committed) | Pairs |
|---|---|---|---|
| Upworthy Research Archive | https://osf.io/jd64p/ | `upworthy-archive-datasets/upworthy-archive-exploratory-packages-03.12.2020.csv` | 1,929 |
| Reddit submissions (SNAP) | https://metromaps.stanford.edu/data/web-Reddit.html | `data/raw/redditSubmissions.csv.gz` | 500 |

Raw downloads are not committed (see `.gitignore`). Download them to the paths above and rebuild:
```
python upworthy/build_upworthy_pairs.py
python reddit/reddit_pairs.py data/raw/redditSubmissions.csv.gz reddit/
```
Details: [upworthy/README_upworthy.md](upworthy/README_upworthy.md), [reddit/README_reddit.md](reddit/README_reddit.md).

Split reproduction, test-set hashes, and independent data checks:
[xian_data_prep/README.md](xian_data_prep/README.md).

The active split contains 386 development pairs and 1,543 test pairs, with zero
shared normalized headlines or image IDs. Excerpts are excluded from grouping
and model inputs. The previous headline-only split is archived in
`xian_data_prep/archive/headline-v1/`.

## Rules
- The test set is locked after the dev/test split (hash recorded). Prompt tuning and thresholds use dev only.
- Reported numbers come from the merged results only.

## Model experiment protocol

[config.yaml](config.yaml), [prompts/](prompts/), and
[few_shot.json](few_shot.json) define the shared Qwen protocol and four fixed dev
examples. [EXPERIMENT_LOCK.json](EXPERIMENT_LOCK.json) records the frozen files.
The dev scoring pool contains 379 pairs after example and near-duplicate
exclusions. See [prompts/README.md](prompts/README.md) for validation, model digest,
sampling settings and run commands. Offline checks passed; the live dev pilot
remains pending on a machine with Ollama and the pinned model.
