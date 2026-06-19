"""
Train TabTransformer on CIC-UNSW-NB15 (Mac CPU/MPS)
====================================================
Treats each of 76 features as a token in a sequence,
uses Transformer self-attention to learn feature interactions,
then MLP head for 10-class classification.

Architecture:
  Feature embedding (76 × 32-dim)
  └→ Positional encoding
     └→ Transformer Encoder × 4 (4 heads)
        └→ Mean pooling
           └→ MLP head (32 → 16 → 10)

SMOTE + class-weighted loss for imbalance.
Data: data/processed/split/
Output: models_saved/tabtransformer_model.pth
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import time, os

# ── Device ──
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal GPU on Mac)")
else:
    device = torch.device("cpu")
    print("Using CPU")
torch.set_num_threads(os.cpu_count())

# ── Paths ──
DATA_DIR = "data/processed/split"
OUT_DIR  = "models_saved"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "Benign", "Analysis", "Backdoor", "DoS", "Exploits",
    "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
]

# ── Load data ──
X_train = np.load(f"{DATA_DIR}/X_train_scale.npy").astype(np.float32)
X_val   = np.load(f"{DATA_DIR}/X_val_scale.npy").astype(np.float32)
X_test  = np.load(f"{DATA_DIR}/X_test_scale.npy").astype(np.float32)
y_train = np.load(f"{DATA_DIR}/y_train.npy").flatten().astype(np.int64)
y_val   = np.load(f"{DATA_DIR}/y_val.npy").flatten().astype(np.int64)
y_test  = np.load(f"{DATA_DIR}/y_test.npy").flatten().astype(np.int64)

n_features   = X_train.shape[1]   # 76
n_classes    = len(np.unique(y_train))  # 10
SEQ_LENGTH   = n_features         # 76 tokens (one per feature)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Features: {n_features}, Classes: {n_classes}")

# ── SMOTE ──
class_counts = np.bincount(y_train.astype(int))
target_per_class = int(class_counts.max() * 0.1)

sampling_strategy = {}
for i, count in enumerate(class_counts):
    if count < target_per_class:
        sampling_strategy[i] = target_per_class

print(f"\nSMOTE target: {target_per_class:,}/class")
print(f"Classes to oversample: {list(sampling_strategy.keys())}")

smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=5, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_smote.shape}")

# ── Class weights ──
n_smote = len(y_train_smote)
smote_counts = np.bincount(y_train_smote.astype(int))
class_weights = n_smote / (n_classes * smote_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("\nClass weights:")
for i, w in enumerate(class_weights):
    print(f"  {CLASS_NAMES[i]:<15} {w:.4f}")

# ── Model ──
class TabTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d_model=32, nhead=4, n_layers=4, d_ff=128, dropout=0.2):
        super().__init__()
        self.n_features = n_features

        # Step 1: embed each of the 76 features into a d_model-dim vector
        self.feature_embedding = nn.Linear(1, d_model)

        # Step 2: learnable positional encoding (feature index 0..75)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)

        # Step 3: Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,  # Pre-norm: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Step 4: MLP head (pooled vector → 10 classes)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x):
        # x: (batch, n_features) = (B, 76)
        # Reshape to (B, 76, 1) — each feature as its own "token"
        x = x.unsqueeze(-1)

        # Embed each feature: (B, 76, 1) → (B, 76, d_model)
        x = self.feature_embedding(x)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer Encoder: (B, 76, d_model) → (B, 76, d_model)
        x = self.transformer(x)

        # Mean pool over all features: (B, 76, d_model) → (B, d_model)
        x = x.mean(dim=1)

        # Head: (B, d_model) → (B, n_classes)
        x = self.head(x)
        return x


D_MODEL  = 32   # embedding dimension
NHEAD    = 4    # attention heads
N_LAYERS = 4    # transformer layers
D_FF     = 128  # feed-forward hidden dim

model = TabTransformer(
    n_features=n_features,
    n_classes=n_classes,
    d_model=D_MODEL,
    nhead=NHEAD,
    n_layers=N_LAYERS,
    d_ff=D_FF,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {total_params:,} parameters")
print(model)

# ── DataLoaders ──
BATCH_SIZE = 512  # smaller than MLP (Transformer uses more memory)

train_dataset = TensorDataset(torch.from_numpy(X_train_smote),
                               torch.from_numpy(y_train_smote))
val_dataset   = TensorDataset(torch.from_numpy(X_val),
                               torch.from_numpy(y_val))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Optimizer & Scheduler ──
criterion   = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer   = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

# ── Training loop ──
EPOCHS   = 150
PATIENCE = 15

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
plt.savefig(f"{OUT_DIR}/tabtransformer_training_curve.png", dpi=150)
plt.show()

# ── Evaluate on test set ──
model.load_state_dict(best_state)
model.eval()

test_dataset = TensorDataset(torch.from_numpy(X_test),
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
print("TEST SET RESULTS")
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
plt.title("Confusion Matrix (Test Set)")
thresh = cm.max() / 2
for i in range(n_classes):
    for j in range(n_classes):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/tabtransformer_confusion_matrix.png", dpi=150)
plt.show()

# ── Save model ──
save_path = f"{OUT_DIR}/tabtransformer_model.pth"
torch.save({
    "model_state_dict": best_state,
    "n_features": n_features,
    "n_classes": n_classes,
    "d_model": D_MODEL,
    "nhead": NHEAD,
    "n_layers": N_LAYERS,
    "d_ff": D_FF,
    "class_names": CLASS_NAMES,
    "class_weights": class_weights.tolist(),
    "smote": True,
    "smote_strategy": float(target_per_class),
    "test_macro_f1": float(macro_f1),
    "test_weighted_f1": float(weighted_f1),
}, save_path)
print(f"Model saved to {save_path}")

# ── Summary ──
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Architecture: TabTransformer (d_model={D_MODEL}, nhead={NHEAD}, layers={N_LAYERS})")
print(f"Parameters:   {total_params:,}")
print(f"SMOTE:        target={target_per_class:,}/class, k=5")
print(f"  Original:   {len(X_train):,} → {len(X_train_smote):,}")
print(f"Batch size:   {BATCH_SIZE}")
print(f"Optimizer:    AdamW (lr=5e-4, wd=1e-4)")
print(f"Scheduler:    CosineAnnealingLR")
print(f"Loss:         CrossEntropy (class-weighted + SMOTE)")
print(f"Early stop:   patience={PATIENCE}")
print(f"Training:     {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"Test Macro F1:   {macro_f1:.4f}")
print(f"Test Weighted F1: {weighted_f1:.4f}")
