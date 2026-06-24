# Model Comparison & Deployment Decision — CIC-UNSW-NB15

## Final Model Comparison (67,188 test samples)

| Rank | Model | Macro F1 | Weighted F1 | Notes |
|---|---|---|---|---|
| 1 | **XGBoost tuned** | **0.6292** | 0.9333 | Best overall — selected for deployment |
| 2 | LightGBM tuned | 0.6244 | **0.9341** | Close second, slightly better Weighted F1 |
| 3 | XGBoost + interaction features | 0.6058 | 0.9309 | Feature engineering added noise |
| 4 | XGBoost baseline | 0.6014 | 0.9313 | Original params |
| 5 | A3 Weighted Voting (XGB+TT) | 0.6047 | 0.9328 | Tiny gain, not worth complexity |
| 6 | Original Hybrid (logits+TT) | 0.5786 | 0.9275 | Outperformed by XGBoost alone |
| 7 | LightGBM baseline | 0.5566 | 0.9287 | |
| 8 | Joint training (A2) | 0.5504 | 0.9230 | |
| 9 | CatBoost tuned | 0.5325 | 0.9255 | Worst tree model on this data |
| 10 | CatBoost baseline | 0.4965 | 0.9206 | |
| 11 | TabTransformer | 0.4757 | 0.9132 | Deep learning lags on tabular data |
| 12 | MLP | 0.4422 | 0.9011 | Simple neural baseline |

## Deployment Decision

**Model:** XGBoost tuned (`models_saved/xgboost_best.json`)

**Best hyperparameters:**
```json
{
  "n_estimators": 1000,
  "max_depth": 10,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "min_child_weight": 5,
  "gamma": 0,
  "reg_alpha": 1.0
}
```

**Rationale:**
- Highest Macro F1 (0.6292) — most important for imbalanced classes
- No external dependencies (unlike LightGBM/CatBoost)
- Fast inference (~1ms per sample)
- Well-calibrated confidence (avg 0.97 when correct vs 0.49 when wrong)

**Per-class performance of deployed model:**

| Class | F1-Score | Support | Note |
|---|---|---|---|
| Benign | 0.9887 | 53,750 | Near-perfect |
| Exploits | 0.7739 | 4,643 | |
| Fuzzers | 0.7221 | 4,442 | |
| Reconnaissance | 0.6950 | 2,510 | |
| Backdoor | 0.5736 | 68 | Rare |
| Generic | 0.7491 | 695 | |
| DoS | 0.4429 | 670 | |
| Worms | 0.5500 | 37 | Extremely rare (0.05%) |
| Shellcode | 0.3822 | 315 | |
| Analysis | 0.4146 | 58 | Rare |

**Key takeaway:** The model is excellent on majority classes but struggles on minority classes with <100 samples. This is expected given the severe class imbalance.

## Next Steps
1. Build FastAPI deployment with XGBoost tuned model
2. Demo on real web traffic
3. (Optional) Further improve rare-class recall via data augmentation or cost-sensitive learning
