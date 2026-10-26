# Upworthy Dataset Section — Data Documentation & Processing Pipeline
**Author**: Yurun Shi (ECS 189G)  
**Deliverable**: Task 4 Data README Section  
**File**: `upworthy_pairs.csv`  

---

## 1. Overview & Raw Data Source
* **Source Archive**: The public exploratory subset of the Upworthy Research Archive ([Matias et al., 2021](https://osf.io/jd64p/)).
* **Raw File**: `upworthy-archive-datasets/upworthy-archive-exploratory-packages-03.12.2020.csv`
* **Raw Size**: 22,666 packages across 4,873 unique tests (`clickability_test_id`).

---

## 2. Filtering Pipeline & Counts Breakdown

| Stage | Description | Tests Remaining | Packages / Pairs |
| :--- | :--- | :--- | :--- |
| **0. Raw Exploratory** | All tests in the public exploratory CSV. | 4,873 tests | 22,666 packages |
| **1. Unreliable Randomization** | Filter out tests created between `2013-06-25` and `2014-01-10` due to the Cloudflare caching glitch documented in the 2024 update. | 3,823 tests (-21.5%) | 18,388 packages |
| **2. Causal Pair Isolation** | Require candidate package pairs within a test to share identical `eyecatcher_id` (image), identical `excerpt`, distinct headlines, and `impressions >= 500`. | 1,929 tests | 16,486 candidate pairs |
| **3. Deterministic Sampling** | Sample exactly 1 pair per test with `random.seed(42)` and coin-flip A/B position assignment. Drop exact CTR ties (1 pair dropped). | **1,929 tests** | **1,929 final pairs** |

---

## 3. Dataset Columns in `upworthy_pairs.csv`

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `test_id` | `str` | Unique Upworthy experiment identifier (`clickability_test_id`). |
| `created_at` | `str` | Timestamp of experiment creation. |
| `eyecatcher_id` | `str` | Image ID common to both packages (guarantees image visual invariance). |
| `excerpt` | `str` | Blurb text common to both packages (guarantees excerpt invariance). |
| `headline_A` | `str` | Headline assigned as Option A (random coin-flip assigned). |
| `headline_B` | `str` | Headline assigned as Option B. |
| `impressions_A` | `int` | Number of times Package A was shown to visitors. |
| `clicks_A` | `int` | Number of times Package A was clicked. |
| `ctr_A` | `float` | Observed Click-Through Rate for A: `clicks_A / impressions_A`. |
| `impressions_B` | `int` | Number of times Package B was shown to visitors. |
| `clicks_B` | `int` | Number of times Package B was clicked. |
| `ctr_B` | `float` | Observed Click-Through Rate for B: `clicks_B / impressions_B`. |
| `true_winner` | `str` | Ground truth label (`"A"` or `"B"`). Strictly higher CTR. |
| `delta_ctr` | `float` | Difference in CTR (`ctr_A - ctr_B`). |
| `abs_relative_diff`| `float` | Relative CTR difference: `abs(ctr_A - ctr_B) / max(ctr_A, ctr_B)`. |
| `z_score` | `float` | Two-proportion pooled z-score test comparing CTR_A and CTR_B. |
| `p_value` | `float` | Two-tailed p-value for the difference in proportions. |
| `is_significant` | `bool` | `True` if `p_value < 0.05` ($|z| \ge 1.96$), `False` otherwise (near-tie/noisy). |

---

## 4. Summary Statistics & Sanity Checks
* **Sample Size**: 1,929 pairs.
* **Position Balance**:
  * Winner = A: 940 pairs (48.73%)
  * Winner = B: 989 pairs (51.27%)
  * *Confirmation: Position assignment is approximately balanced via coin-flip randomization.*
* **Statistical Significance**:
  * Significant ($p < 0.05$): **551 pairs (28.6%)**
  * Non-significant ($p \ge 0.05$): **1,378 pairs (71.4%)**
  * *Matches the proposal plan: full 1,929 population for RQ2 skipping policy; 551 significant subset for RQ1 diagnostic accuracy.*
* **Impression Safety**:
  * Minimum impressions for any package in `upworthy_pairs.csv` is 758 (well above the 500 threshold). Median package impressions: ~3,100.
