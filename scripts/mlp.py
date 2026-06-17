import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import time, os

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

torch.set_num_threads(os.cpu_count())

DATA_DIR = "../data/processed/split"
OUT_DIR = "models_saved"
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES_NAMES = [
    "Benign",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]

X_train = np.load(f"{DATA_DIR}/X_train_scale.npy").astype(np.float32)
X_val = np.load(f"{DATA_DIR}/X_val_scale.npy").astype(np.float32)
X_test = np.load(f"{DATA_DIR}/X_test_scale.npy").astype(np.float32)
y_train = np.load(f"{DATA_DIR}/y_train.npy").flatten().astype(np.int64)
y_val = np.load(f"{DATA_DIR}/y_val.npy").flatten().astype(np.int64)
y_test = np.load(f"{DATA_DIR}/y_test.npy").flatten().astype(np.int64)

n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))
print(f"Before SMOTE - Train : {X_train.shape}, Val : {X_val.shape}, Test : {X_test.shape}")


class_counts = np.bincount(y_train.astype(int))
target_per_class = int(class_counts.max() * 0.1)

sampling_strategy = {}
for i, count in enumerate(class_counts):
    if count < target_per_class:
        sampling_strategy[i] = target_per_class

print(f"Smote target per class: {target_per_class}")
print(f"Classes to oversample : {list(sampling_strategy.keys())}")

print("Applying SMOTE to balance the training dataset")
smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=5, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Train : {X_train_smote.shape}")

n_smote = len(y_train_smote)
smote_counts = np.bincount(y_train_smote.astype(int))
class_weights = n_smote / (n_classes * smote_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print(f"Class weights: {class_weights}")
for i, w in enumerate(class_weights):
    print(f"Class {i} weight: {w:.4f}")



class MLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):
        return self.net(x)

model = MLP(n_features, n_classes).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel : {total_params:,} parameters")

# Dataloaders
BATCH_SIZE = 4096
train_dataset = TensorDataset(torch.from_numpy(X_train_smote), torch.from_numpy(y_train_smote))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Optimizer & Scheduler
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=2e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training loop
EPOCHS = 3000
PATIENCE = 100

train_losses = []
val_losses = []
best_val_loss = float("inf")
best_state = None 
patience_counter = 0
start = time.time()
epoch_train_loss = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # grad = grad * 5 / norm ; 1/2048 -> gradient cao 
        optimizer.step()
        epoch_train_loss += loss.item() * Xb.size(0)

    scheduler.step()
    epoch_train_loss /= len(train_loader.dataset)
    

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
    else : 
        patience_counter += 1
    

    if epoch % 10 == 0 or epoch == 1:
        elapsed_time = time.time() - start
        print(f"Epoch [{epoch}/{EPOCHS}] - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Time: {elapsed_time:.2f}s")

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
        break
total_time = time.time() - start

print(f"Done in {total_time:.2f}s. Best Val Loss: {best_val_loss:.4f}")

# Ploting 
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.axvline(np.argmin(val_losses), color="r", linestyle="--", label="Best Val Loss", alpha=0.5)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUT_DIR}/mlp_training_curve.png", dpi=150)
plt.show()


# Eval
model.load_state_dict(best_state)
model.eval()

test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_preds = []
all_labels = []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        outputs = model(Xb)
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

macro_f1 = f1_score(all_labels, all_preds, average="macro")
weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
print("Test set results:")
print(f"Macro F1 Score: {macro_f1:.4f}")
print(f"Weighted F1 Score: {weighted_f1:.4f}")
print()
print(classification_report(all_labels, all_preds, target_names=CLASSES_NAMES, digits=4))


# # Confusion Matrix
# cm = confusion_matrix(all_labels, all_preds)
# plt.figure(figsize=(12, 10))
# plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
# plt.colorbar()
# plt.xticks(range(n_classes), CLASSES_NAMES, rotation=45, ha="right")
# plt.yticks(range(n_classes), CLASSES_NAMES)
# plt.xlabel("Predicted label")
# plt.ylabel("True label")
# plt.tile("Confusion Matrix")
# thresh = cm.max() / 2.0
# for i in range(n_classes):
#     for j in range(n_classes):
#         plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=8)

# plt.tight_layout()
# plt.savefig(f"{OUT_DIR}/mlp_confusion_matrix.png", dpi=150)
# plt.show()

# Save model 
save_path = f"{OUT_DIR}/mlp_model.pth"
torch.save({
    "model_state_dict": best_state,
    "n_features": n_features,
    "n_classes": n_classes,
    "class_weights" : class_weights.tolist(),
    "smote" : True,
    "smote_strategy" : float(target_per_class),
    "test_macro_f1" : float(macro_f1),
    "test_weighted_f1" : float(weighted_f1),
}, save_path)
print(f"Model saved to {save_path}")


# Summary 
print("\nSummary:")
print(f"Total parameters: {total_params:,}")
print("Architecture: MLP with 3 hidden layers (512, 256, 128 units) with BatchNorm and Dropout")
print("SMOTE target per class:", target_per_class)
print(f"Original : {len(X_train):,} samples, After SMOTE : {len(X_train_smote):,} samples")
print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, Patience: {PATIENCE}")
print(f"training {total_time:.2f}s, Best Val Loss: {best_val_loss:.4f}, Test Macro F1: {macro_f1:.4f}, Test Weighted F1: {weighted_f1:.4f}")