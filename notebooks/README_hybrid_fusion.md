# Hybrid Fusion — XGBoost + TabTransformer

## Purpose

Combine **XGBoost** (decision trees) and **TabTransformer** (self-attention) into a single model, aiming to outperform either one alone.

## How It Works

**Step 1** — Load 2 pre-trained models:
- `xgboost_v1.json` (from XGBoost training)
- `tabtransformer_model.pth` (from TabTransformer training)

**Step 2** — Extract embeddings from each model:
- **XGBoost**: raw logits (before softmax) — `shape (n, 10)`
- **TabTransformer**: remove head, get 32-dim pooled embedding — `shape (n, 32)`

**Step 3** — Concatenate: `10 + 32 = 42 features per sample`

**Step 4** — Train a small Fusion MLP:
```
Input (42 features)
  → Linear(42 → 32) + ReLU + Dropout
  → Linear(32 → 16) + ReLU
  → Linear(16 → 10 classes)
```

The Fusion MLP learns to weigh each model's opinion: trust XGBoost more for some classes, trust Transformer more for others.

## Files

| File | Description |
|------|-------------|
| `hybrid_fusion.ipynb` | Notebook to run on Colab GPU |
| `models_saved/hybrid_fusion.pth` | Trained fusion model |

## How to Run (Google Colab)

1. Upload these files to Google Drive `Colab Notebooks/anomaly-data/`:
   - `data/processed/split/*.npy` (6 files: X/y train/val/test, scaled + unscaled)
   - `models_saved/xgboost_v1.json`
   - `models_saved/tabtransformer_model.pth`

2. Open `hybrid_fusion.ipynb` in Colab
3. Runtime → Change runtime type → **T4 GPU**
4. Run all cells

## Expected Results

| Model | Macro F1 | Weighted F1 |
|-------|----------|-------------|
| XGBoost | 0.6014 | 0.9313 |
| MLP | 0.4422 | 0.9011 |
| TabTransformer | 0.4757 | 0.9132 |
| **Hybrid Fusion** | **?** | **?** |

If Hybrid > 0.60 Macro F1 → fusion works. If ≤ 0.60 → XGBoost is still the best model, deploy XGBoost instead.
