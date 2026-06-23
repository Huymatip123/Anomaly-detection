"""
Hybrid Fusion: XGBoost + TabTransformer
=========================================
Extracts:
  - XGBoost raw logits (10-dim, before softmax)
  - TabTransformer pooled embedding (32-dim, before head)
Concatenates → 42-dim → Fusion MLP → 10 classes.

Strategy: Two-stage (freeze both base models, train only fusion MLP).
Data: data/processed/split/
Output: models_saved/hybrid_fusion.pth
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import time, os

# ── Device ──
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ── Paths ──
DATA_DIR = "data/processed/split"
OUT_DIR  = "models_saved"

CLASS_NAMES = [
    "Benign", "Analysis", "Backdoor", "DoS", "Exploits",
    "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
]

# ── Load data ──
X_train = np.load(f"{DATA_DIR}/X_train.npy").astype(np.float32)
X_val   = np.load(f"{DATA_DIR}/X_val.npy").astype(np.float32)
X_test  = np.load(f"{DATA_DIR}/X_test.npy").astype(np.float32)
X_train_scale = np.load(f"{DATA_DIR}/X_train_scale.npy").astype(np.float32)
X_val_scale   = np.load(f"{DATA_DIR}/X_val_scale.npy").astype(np.float32)
X_test_scale  = np.load(f"{DATA_DIR}/X_test_scale.npy").astype(np.float32)
y_train = np.load(f"{DATA_DIR}/y_train.npy").flatten().astype(np.int64)
y_val   = np.load(f"{DATA_DIR}/y_val.npy").flatten().astype(np.int64)
y_test  = np.load(f"{DATA_DIR}/y_test.npy").flatten().astype(np.int64)

n_features = X_train.shape[1]
n_classes  = 10

# ═══════════════════════════════════════════════════════════════
# 1. Load pre-trained XGBoost → extract raw logits (10-dim)
# ═══════════════════════════════════════════════════════════════
print("\n=== XGBoost: extracting logits ===")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(f"{OUT_DIR}/xgboost_v1.json")

# output_margin=True gives raw logits before softmax → shape (n, 10)
train_xgb_logits = xgb_model.predict(X_train, output_margin=True).astype(np.float32)
val_xgb_logits   = xgb_model.predict(X_val,   output_margin=True).astype(np.float32)
test_xgb_logits  = xgb_model.predict(X_test,  output_margin=True).astype(np.float32)
print(f"XGBoost logits — train: {train_xgb_logits.shape}, val: {val_xgb_logits.shape}, test: {test_xgb_logits.shape}")

# ═══════════════════════════════════════════════════════════════
# 2. Load pre-trained TabTransformer → extract pooled embedding (32-dim)
# ═══════════════════════════════════════════════════════════════
print("\n=== TabTransformer: extracting embeddings ===")

D_MODEL  = 32
NHEAD    = 4
N_LAYERS = 4
D_FF     = 128

class TabTransformerEmbed(nn.Module):
    """Same as TabTransformer but returns pooled embedding instead of logits."""
    def __init__(self, n_features, d_model=32, nhead=4, n_layers=4, d_ff=128):
        super().__init__()
        self.feature_embedding = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=0.2, activation="relu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        return x.mean(dim=1)  # (B, d_model) — pooled embedding

# Load saved TabTransformer weights into embed-only version
saved = torch.load(f"{OUT_DIR}/tabtransformer_model.pth", map_location="cpu", weights_only=False)
state_dict = saved["model_state_dict"]

# Filter out head layers (we don't need them)
embed_state_dict = {k: v for k, v in state_dict.items() if k.startswith("feature_embedding") or k.startswith("pos_encoding") or k.startswith("transformer")}

tt_embed = TabTransformerEmbed(n_features, D_MODEL, NHEAD, N_LAYERS, D_FF)
tt_embed.load_state_dict(embed_state_dict, strict=False)
tt_embed.to(device)
tt_embed.eval()

# Extract embeddings (no gradients needed — frozen model)
print("Extracting train embeddings...")
train_tt_embed = []
with torch.no_grad():
    for i in range(0, len(X_train_scale), 2048):
        batch = torch.from_numpy(X_train_scale[i:i+2048]).to(device)
        train_tt_embed.append(tt_embed(batch).cpu().numpy())
train_tt_embed = np.concatenate(train_tt_embed, axis=0).astype(np.float32)

print("Extracting val embeddings...")
val_tt_embed = []
with torch.no_grad():
    for i in range(0, len(X_val_scale), 2048):
        batch = torch.from_numpy(X_val_scale[i:i+2048]).to(device)
        val_tt_embed.append(tt_embed(batch).cpu().numpy())
val_tt_embed = np.concatenate(val_tt_embed, axis=0).astype(np.float32)

print("Extracting test embeddings...")
test_tt_embed = []
with torch.no_grad():
    for i in range(0, len(X_test_scale), 2048):
        batch = torch.from_numpy(X_test_scale[i:i+2048]).to(device)
        test_tt_embed.append(tt_embed(batch).cpu().numpy())
test_tt_embed = np.concatenate(test_tt_embed, axis=0).astype(np.float32)

print(f"TabTransformer embed — train: {train_tt_embed.shape}, val: {val_tt_embed.shape}, test: {test_tt_embed.shape}")

# ═══════════════════════════════════════════════════════════════
# 3. Concatenate embeddings → Fusion MLP
# ═══════════════════════════════════════════════════════════════
print("\n=== Fusion MLP ===")

X_train_fusion = np.concatenate([train_xgb_logits, train_tt_embed], axis=1)  # (n, 42)
X_val_fusion   = np.concatenate([val_xgb_logits,   val_tt_embed],   axis=1)  # (n, 42)
X_test_fusion  = np.concatenate([test_xgb_logits,  test_tt_embed],  axis=1)  # (n, 42)

fusion_dim = X_train_fusion.shape[1]
print(f"Fusion input dim: {fusion_dim} (10 XGBoost + {D_MODEL} Transformer)")

# Class weights for fusion MLP
class_counts = np.bincount(y_train.astype(int))
n_smote = len(y_train)
class_weights = n_smote / (n_classes * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Fusion MLP
class FusionMLP(nn.Module):
    def __init__(self, fusion_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.net(x)

model = FusionMLP(fusion_dim, n_classes).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Fusion MLP params: {total_params:,}")

# DataLoaders
BATCH_SIZE = 2048
train_dataset = TensorDataset(torch.from_numpy(X_train_fusion),
                               torch.from_numpy(y_train))
val_dataset   = TensorDataset(torch.from_numpy(X_val_fusion),
                               torch.from_numpy(y_val))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ── Training loop ──
EPOCHS = 100
PATIENCE = 10

train_losses = []
val_losses = []
best_val_loss = float("inf")
best_state = None
patience_counter = 0
start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_train_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        epoch_train_loss += loss.item() * Xb.size(0)

    scheduler.step()
    epoch_train_loss /= len(train_dataset)

    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            epoch_val_loss += loss.item() * Xb.size(0)
    epoch_val_loss /= len(val_dataset)

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - start
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

    if patience_counter >= PATIENCE:
        print(f"→ Early stopping at epoch {epoch}")
        break

total_time = time.time() - start
print(f"\nDone in {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"Best val loss: {best_val_loss:.4f}")

# ── Plot ──
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.axvline(np.argmin(val_losses), color="red", ls="--", alpha=0.5, label="Best")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUT_DIR}/hybrid_training_curve.png", dpi=150)
plt.show()

# ── Evaluate on test set ──
model.load_state_dict(best_state)
model.eval()

test_dataset = TensorDataset(torch.from_numpy(X_test_fusion),
                              torch.from_numpy(y_test))
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_preds = []
all_labels = []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        outputs = model(Xb)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

macro_f1     = f1_score(all_labels, all_preds, average="macro")
weighted_f1  = f1_score(all_labels, all_preds, average="weighted")

print("=" * 60)
print("HYBRID FUSION TEST SET RESULTS")
print("=" * 60)
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print()
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

# ── Confusion matrix ──
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar()
plt.xticks(range(n_classes), CLASS_NAMES, rotation=45, ha="right")
plt.yticks(range(n_classes), CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Hybrid Fusion)")
thresh = cm.max() / 2
for i in range(n_classes):
    for j in range(n_classes):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/hybrid_confusion_matrix.png", dpi=150)
plt.show()

# ── Save model ──
save_path = f"{OUT_DIR}/hybrid_fusion.pth"
torch.save({
    "model_state_dict": best_state,
    "fusion_dim": fusion_dim,
    "n_classes": n_classes,
    "class_names": CLASS_NAMES,
    "test_macro_f1": float(macro_f1),
    "test_weighted_f1": float(weighted_f1),
}, save_path)
print(f"Model saved to {save_path}")

# ── Final comparison ──
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print(f"{'Model':<20} {'Macro F1':>10} {'Weighted F1':>12}")
print("-" * 44)
print(f"{'XGBoost':<20} {0.6014:>10.4f} {0.9313:>12.4f}")
print(f"{'MLP':<20} {0.4422:>10.4f} {0.9011:>12.4f}")
print(f"{'TabTransformer':<20} {0.4757:>10.4f} {0.9132:>12.4f}")
print(f"{'Hybrid Fusion':<20} {macro_f1:>10.4f} {weighted_f1:>12.4f}")
