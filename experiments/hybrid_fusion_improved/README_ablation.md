# Ablation Study: Hybrid Fusion (XGBoost + TabTransformer)

This ablation study investigates whether combining XGBoost and TabTransformer
improves classification on the **CIC-UNSW-NB15** dataset — a highly imbalanced
(80% Benign, 0.05% Worms) multi-class network intrusion benchmark with 76
numerical features.

## Why this dataset?

CIC-UNSW-NB15 is **tabular data**. A well-known NeurIPS 2022 paper
("Why do tree-based models still outperform deep learning on tabular data?")
demonstrates that gradient-boosted trees consistently beat neural networks on
this class of problems. Our own experiments confirm this: XGBoost achieves
**0.6014 Macro F1** while the Transformer alone gets **0.4757**.

The question is not "which model wins?" — XGBoost clearly does.
The question is: **can the Transformer add any complementary signal that
XGBoost alone misses?**

## Ablation tests and rationale

### 1. Individual model performance
**Why:** Establish baselines. If one model dominates completely, fusion is
pointless.

### 2. Fixed-weight fusion strategies (Ablation 2)
Tests different fixed rules for combining probabilities, with **no training**.
If any fixed rule beats XGBoost, then TT adds signal. If none do, the models
are redundant.

| Strategy | Rationale |
|---|---|
| **Simple average** | Baseline fusion — are the models complementary at all? |
| **Max confidence** | Trust the model that's more sure — does confidence = accuracy? |
| **Geometric mean** | Conservative: penalizes low probabilities heavily |
| **Harmonic mean** | Even more conservative than geometric |
| **XGBoost-first (threshold)** | Use TT only when XGBoost is uncertain — mimics real deployment |

### 3. Per-class analysis (Ablation 3)
**Why:** Even if XGBoost wins overall, TT might be better on rare classes.
The dataset is severely imbalanced, so per-class breakdown reveals where
each model struggles.

### 4. Disagreement analysis (Ablation 4)
**Why:** Fusion only helps when models disagree AND one is correct. Counts
disagreements and measures which model wins on those samples. The **oracle**
(always picking the correct model) gives the upper bound of improvement.

### 5. Learned weights / Weighted Voting — re-trained (Ablation 5)
**Why:** The original A3 (trained on Colab) reported **0.6031 Macro F1**,
slightly above XGBoost's **0.6014**. But the learned weights are ~100%
XGBoost for all classes. This is a contradiction:
- If weights are 100% XGBoost, A3 predictions should = XGBoost predictions
- Yet A3 scores 0.6031 vs 0.6014

This ablation re-trains A3 with **multiple random seeds** to check if the
improvement is reproducible or just numerical noise (floating point rounding
in softmax, differences in argmax tie-breaking, etc.).

### 6. Upper bounds (Ablation 6)
**Why:** The **oracle** (pick the correct model per sample) shows the maximum
possible improvement from fusion. If the oracle is only marginally above
XGBoost, then no fusion method can significantly improve results.

### 7. Confidence analysis (Ablation 7)
**Why:** Are the models well-calibrated? Does high confidence mean correct?
If XGBoost is confident when wrong and TT is uncertain when right, confidence-
based fusion (like XGBoost-first) won't work well.

## Expected outcomes

| Finding | Interpretation |
|---|---|
| Fixed fusion ≈ XGBoost | TT adds no signal |
| Fixed fusion > XGBoost | TT captures patterns XGBoost misses |
| A3 0.6031 not reproducible | Previous result was floating-point noise |
| A3 0.6031 reproducible | A3 genuinely improves, but by tiny margin |
| Oracle >> XGBoost | Fusion has room to improve |
| Oracle ≈ XGBoost | Models make same errors — fusion is futile |
