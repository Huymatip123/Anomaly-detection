# Hybrid Fusion — XGBoost + TabTransformer

## Mục đích

Kết hợp **XGBoost** (cây quyết định) và **TabTransformer** (self-attention) thành 1 model duy nhất, hy vọng mạnh hơn từng cái riêng lẻ.

## Cách hoạt động

**Bước 1** — Load 2 model đã train:
- `xgboost_v1.json` (từ XGBoost training)
- `tabtransformer_model.pth` (từ TabTransformer training)

**Bước 2** — Trích xuất embedding từ mỗi model:
- **XGBoost**: lấy 10 logits (giá trị trước softmax) — `shape (n, 10)`
- **TabTransformer**: bỏ head, lấy pooled embedding 32 chiều — `shape (n, 32)`

**Bước 3** — Ghép nối: `10 + 32 = 42 features / 1 mẫu`

**Bước 4** — Train Fusion MLP nhỏ:
```
Input (42 features)
  → Linear(42 → 32) + ReLU + Dropout
  → Linear(32 → 16) + ReLU
  → Linear(16 → 10 classes)
```

Fusion MLP học cách kết hợp ý kiến của cả 2 model: tin XGBoost hơn cho class này, tin Transformer hơn cho class kia.

## File

| File | Mô tả |
|------|-------|
| `hybrid_fusion.ipynb` | Notebook chạy trên Colab GPU |
| `models_saved/hybrid_fusion.pth` | Model sau khi train |

## Cách chạy (Google Colab)

1. Upload các file này lên Google Drive `Colab Notebooks/anomaly-data/`:
   - `data/processed/split/*.npy` (6 files: X/y train/val/test, cả scaled + unscaled)
   - `models_saved/xgboost_v1.json`
   - `models_saved/tabtransformer_model.pth`

2. Mở `hybrid_fusion.ipynb` trong Colab
3. Runtime → Change runtime type → **T4 GPU**
4. Run all cells

## Kết quả kỳ vọng

| Model | Macro F1 | Weighted F1 |
|-------|----------|-------------|
| XGBoost | 0.6014 | 0.9313 |
| MLP | 0.4422 | 0.9011 |
| TabTransformer | 0.4757 | 0.9132 |
| **Hybrid Fusion** | **?** | **?** |

Nếu Hybrid > 0.60 Macro F1 → kết hợp thành công. Nếu ≤ 0.60 → XGBoost vẫn là model tốt nhất, dùng XGBoost deploy.
