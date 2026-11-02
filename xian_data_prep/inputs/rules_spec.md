# Experiment & Dataset Rules Specification
**Course**: ECS 189G  
**Project**: Can LLMs Predict A/B Test Results, and Do They Know When They're Wrong?  
**Team**: Yurun Shi, Gaeun Yu, Xian Yang Weng  
**Date**: October 26, 2026 (Task 1 Deliverable)  

---

## 1. Upworthy Dataset Rules

### 1.1 Unreliable Randomization Filter (The 2024 Update)
* **Background**: Matias et al. (2024) uncovered a Cloudflare caching malfunction on Upworthy's platform between **June 25, 2013 and January 10, 2014**. During this window, the cache served a single variant repeatedly rather than randomly allocating traffic across arms, causing severe Sample Ratio Mismatches (SRM).
* **Rule**: Any experiment whose `created_at` timestamp falls between **June 25, 2013 00:00:00** and **January 10, 2014 23:59:59** must be dropped.
* **Impact**: Drops 1,050 tests (21.5% of the 4,873 exploratory tests), leaving **3,823 clean tests**.

### 1.2 Pair Eligibility & Causal Isolation Rule
To ensure that any difference in Click-Through Rate (CTR) is strictly caused by the headline text and not visual or blurb confounders:
* Two packages in the same test (`clickability_test_id`) form an eligible pair if and only if:
  1. **Identical Image**: `eyecatcher_id` is identical and non-empty.
  2. **Identical Excerpt**: `excerpt` text is identical.
  3. **Distinct Headline**: Normalized headlines differ (`headline_1.strip().lower() != headline_2.strip().lower()`).
  4. **Traffic Validity**: Both packages have non-zero traffic (`impressions > 0`).

### 1.3 Minimum Impressions Threshold
* **Empirical Analysis**: In the exploratory archive, package impressions have a median of **3,118** and 99% of packages exceed 1,500 impressions.
* **Rule**: `impressions >= 500` for both packages.
* **Impact**: Only 1 out of 1,929 eligible tests is dropped even at 1,000 impressions. A threshold of 500 protects against early termination noise while preserving 100% of candidate tests (1,929 tests).

### 1.4 Exactly One Pair Per Test (Fixed Seed & Position Neutrality)
* **Independence Rationale**: Multiple pairs from the same test share overlapping audience traffic and baselineCTR; treating them as independent would violate statistical assumptions and artificially inflate sample sizes.
* **Deterministic Pairing Procedure**:
  1. For each test with $\ge 1$ eligible pair, sort all candidate pairs deterministically by package index.
  2. With a fixed seed (`seed = 42`), randomly sample exactly **1 pair** per test using `random.choice`.
  3. **Position Neutrality Coin-Flip**: To prevent bias where the earlier package in the CSV is always assigned to "Headline A", randomly assign the two packages to `headline_A` and `headline_B` via a coin-flip (`random.random() < 0.5`).

### 1.5 Ground Truth & CTR Ties Rule
* **CTR Definition**: $\text{CTR} = \frac{\text{clicks}}{\text{impressions}}$.
* **Winner Assignment**:
  * If $\text{CTR}_A > \text{CTR}_B$, `true_winner = 'A'`.
  * If $\text{CTR}_B > \text{CTR}_A$, `true_winner = 'B'`.
* **Ties**: If $\text{CTR}_A == \text{CTR}_B$, the pair is excluded because binary classification has no objective ground truth.
  * *Empirical Finding*: Out of 16,486 candidate pairs across the archive, only **1 exact tie (0.01%)** exists. Excluding ties has negligible impact on power.
* **Statistical Significance Flag**:
  * Compute a two-proportion pooled z-test:
    $$p_{\text{pool}} = \frac{C_A + C_B}{N_A + N_B}, \quad SE = \sqrt{p_{\text{pool}}(1 - p_{\text{pool}})\left(\frac{1}{N_A} + \frac{1}{N_B}\right)}, \quad z = \frac{\text{CTR}_A - \text{CTR}_B}{SE}$$
  * Record `is_significant = (|z| >= 1.96)` ($\alpha = 0.05$).
  * *All pairs* are kept for the test-skipping policy (RQ2); the significant subset is flagged for diagnostic accuracy (RQ1).

---

## 2. Split Allocation & Fallback Plan Decision (For Xian's Task 9)

* **Eligible Test Count**: 1,929 tests meet all strict criteria (same image + same excerpt).
* **Proposal Contingency**: The proposal stated that if eligible tests fall between 1,500 and 2,500, we **increase the test share and shrink the development split** to ensure at least 1,500 pairs in the locked test split.
* **Agreed Split**:
  * **Development Split**: ~20% (~380 pairs) — used for prompt tuning, baseline training, and setting confidence thresholds.
  * **Locked Test Split**: ~80% (~1,549 pairs) — evaluated strictly once.
* **Leakage-Free Grouping**: All tests must be grouped by normalized headline text (`re.sub(r'[^a-z0-9]', '', text.lower())`) before splitting so no headline appears in both splits.

---

## 3. Reddit Dataset Rules (For Gaeun's Tasks 5 & 6)

### 3.1 Eligibility Criteria
1. **Identical Image Content**: Same `image_id` / URL.
2. **Same Subreddit**: Submissions must occur within the exact same community to hold audience interests constant.
3. **Distinct Titles**: Titles must differ in text.
4. **Time Window**: Submissions must be posted within **30 days** of each other to minimize macro-trend confounding.
5. **Clean Data**: Exclude deleted posts (`[deleted]`), removed posts, or missing values.

### 3.2 Winner Metric & Confounder Tracking
* **Outcome Metric**: `score` (net upvotes = upvotes - downvotes).
* **Winner Assignment**: Title with strictly higher `score`. Exclude exact score ties (`score_A == score_B`).
* **Exposure Bias / First-Post Order**: In Reddit, the earlier post often receives more exposure or is marked as a repost.
  * Gaeun must record:
    * `order`: Which title was posted first (`first_posted`).
    * `time_gap_seconds`: Exact elapsed time between the two submissions.
  * When sampling 1 pair per image with a fixed seed (`seed = 42`), randomly assign title A and title B to preserve position neutrality while retaining the `order` feature for diagnostic analysis.
