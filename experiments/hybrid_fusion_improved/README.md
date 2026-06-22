# Hybrid Fusion Improvements

3 approaches to improve the XGBoost + TabTransformer hybrid fusion model, tested on the CIC-UNSW-NB15 dataset (76 features, 10 classes, ~67K test samples).

## Motivation

The original Hybrid Fusion (concatenate XGBoost logits + TabTransformer embedding → Fusion MLP) scored **Macro F1 = 0.5786**, falling short of XGBoost alone (**0.6014**). This experiment explores 3 strategies to close that gap.

---

## Approach 1: Leaf Indices + PCA

**Goal:** Give the fusion MLP richer information from XGBoost.

**Problem with logits:** XGBoost's 10 raw logits (before softmax) discard most of the tree structure — they only capture the model's final confidence per class, not *why* it arrived at that decision.

**Solution:** Replace 10 logits with leaf indices from all 500 trees. For each sample, XGBoost records which leaf it falls into in every tree → a 500-dimensional vector of integers. These indices encode the full decision path through the forest.

**Pipeline:**
```
500 leaf indices → PCA (32 components) → concat with TT embedding (32) → 64-dim → Fusion MLP
```

**Why PCA?** 500 dimensions is too sparse (each leaf index is a categorical integer). PCA compresses to 32 dense dimensions preserving ~60-80% of the variance.

---

## Approach 2: Joint Training

**Goal:** Let the Transformer adapt to complement XGBoost.

**Problem with frozen TT:** The original Hybrid froze the pre-trained TabTransformer, so it couldn't learn which patterns XGBoost handles poorly → focus on those.

**Solution:** Unfreeze the Transformer and train the entire model end-to-end. Gradients flow from the fusion MLP back through the Transformer encoder, allowing it to shift its representations to cover XGBoost's blind spots.

**Pipeline:**
```
scaled features → Transformer (unfrozen) → embed (32) → concat(XGB logits 10) → 42 → Fusion MLP
```

**Training details:**
- Lower learning rate (1e-4) to avoid destroying pre-trained TT weights
- Smaller batch size (1024) due to memory (Transformer attention)
- Gradient clipping (norm=3.0) for stability

---

## Approach 3: Weighted Soft Voting

**Goal:** The simplest possible fusion with minimal overfitting risk.

**Problem with Fusion MLP:** Thousands of parameters on top of 313K training samples can still overfit, especially when the base models already saturate performance.

**Solution:** Learn just 20 scalar weights — one per class for XGBoost + one per class for Transformer.

```
final_prob[class] = w_xgb[class] × prob_xgb[class] + w_tt[class] × prob_tt[class]

Parameters: 20 floats (2 models × 10 classes)
```

**Why 20 parameters?** With only 20 numbers to learn, overfitting is practically impossible. The weights reveal interpretable information: "which model does the fusion trust more for each attack type?"

Weights are log-transformed and softmax-normalized to stay positive and sum to 1 per class.

---

## Comparison

| Model | Macro F1 | Weighted F1 |
|-------|----------|-------------|
| XGBoost (baseline) | **0.6014** | 0.9313 |
| Original Hybrid | 0.5786 | 0.9275 |
| **1. Leaf PCA + TT** | ? | ? |
| **2. Joint training** | ? | ? |
| **3. Weighted voting** | ? | ? |

---

## How to Run

1. Upload data to Google Drive `Colab Notebooks/anomaly-data/`:
   - `data/processed/split/*.npy` (6 files)
   - `models_saved/xgboost_v1.json`
   - `models_saved/tabtransformer_model.pth`

2. Open `hybrid_fusion_improved.ipynb` in Colab
3. Runtime → Change runtime type → **T4 GPU**
4. Run all cells (~30-60 min depending on GPU load)

The notebook outputs all 3 results + per-class metrics + a final comparison table.
