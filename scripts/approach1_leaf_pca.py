import numpy as np, xgboost as xgb, torch, torch.nn as nn, torch.optim as optim, time, os, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(os.cpu_count())
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device} (torch threads: {torch.get_num_threads()})')

DATA_DIR = 'data/processed/split'
OUT_DIR = 'models_saved'
os.makedirs(OUT_DIR, exist_ok=True)
n_classes = 10
class_names = ['Benign','Analysis','Backdoor','DoS','Exploits',
               'Fuzzers','Generic','Reconnaissance','Shellcode','Worms']

# ---------- Load data ----------
X_train     = np.load(f'{DATA_DIR}/X_train.npy').astype(np.float32)
X_val       = np.load(f'{DATA_DIR}/X_val.npy').astype(np.float32)
X_test      = np.load(f'{DATA_DIR}/X_test.npy').astype(np.float32)
X_train_scale = np.load(f'{DATA_DIR}/X_train_scale.npy').astype(np.float32)
X_val_scale   = np.load(f'{DATA_DIR}/X_val_scale.npy').astype(np.float32)
X_test_scale  = np.load(f'{DATA_DIR}/X_test_scale.npy').astype(np.float32)
y_train = np.load(f'{DATA_DIR}/y_train.npy').flatten().astype(np.int64)
y_val   = np.load(f'{DATA_DIR}/y_val.npy').flatten().astype(np.int64)
y_test  = np.load(f'{DATA_DIR}/y_test.npy').flatten().astype(np.int64)
print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')

# ---------- Load XGBoost ----------
print('Loading XGBoost...')
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(f'{OUT_DIR}/xgboost_v1.json')

# ---------- Fit PCA on subset ----------
print('Fitting PCA(32) on 50K subset...')
t0 = time.time()
subset = xgb_model.apply(X_train[:50000]).astype(np.int32)
pca = PCA(n_components=32, random_state=42)
pca.fit(subset)
print(f'Done {time.time()-t0:.1f}s, explained var: {pca.explained_variance_ratio_.sum():.3f}')

# ---------- Extract leaf PCA in batches ----------
def extract_leaves_pca(X, name):
    print(f'  Leaf PCA {name}...')
    chunk_size = 50000
    results = []
    for i in range(0, len(X), chunk_size):
        chunk = X[i:i+chunk_size]
        leaves = xgb_model.apply(chunk).astype(np.int32)
        results.append(pca.transform(leaves).astype(np.float32))
    return np.concatenate(results, axis=0)

train_leaf_pca = extract_leaves_pca(X_train, 'train')
val_leaf_pca   = extract_leaves_pca(X_val,   'val')
test_leaf_pca  = extract_leaves_pca(X_test,  'test')
print(f'Leaf PCA — train: {train_leaf_pca.shape}')

# ---------- Load TabTransformer ----------
print('Loading TabTransformer...')
D_MODEL, NHEAD, N_LAYERS, D_FF = 32, 4, 4, 128

class TabTransformerFull(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_embedding = nn.Linear(1, D_MODEL)
        self.pos_encoding = nn.Parameter(torch.randn(1, 76, D_MODEL) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=D_FF,
            dropout=0.2, activation='relu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.pool = nn.Sequential(
            nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(D_MODEL // 2, n_classes))

    def forward(self, x, return_embed=False):
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x) + self.pos_encoding
        x = self.transformer(x)
        embed = x.mean(dim=1)
        if return_embed: return embed, self.pool(embed)
        return self.pool(embed)

saved = torch.load(f'{OUT_DIR}/tabtransformer_model.pth', map_location='cpu', weights_only=False)
sd = saved['model_state_dict']
# Rename head -> pool to match TabTransformerFull class
remap = {}
for k, v in sd.items():
    if k.startswith('head.'):
        remap['pool.' + k[5:]] = v
    else:
        remap[k] = v
tt_model = TabTransformerFull()
tt_model.load_state_dict(remap)
tt_model.to(device).eval()

# ---------- Extract TT embeddings ----------
def extract_tt_embed(X, name):
    print(f'  TT {name}...')
    res = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            batch = torch.from_numpy(X[i:i+4096]).to(device)
            embed, _ = tt_model(batch, return_embed=True)
            res.append(embed.cpu().numpy())
    return np.concatenate(res, axis=0).astype(np.float32)

train_tt_embed = extract_tt_embed(X_train_scale, 'train')
val_tt_embed   = extract_tt_embed(X_val_scale,   'val')
test_tt_embed  = extract_tt_embed(X_test_scale,  'test')
print(f'TT embed — train: {train_tt_embed.shape}')

# ---------- Concatenate ----------
X_train_a1 = np.concatenate([train_leaf_pca, train_tt_embed], axis=1).astype(np.float32)
X_val_a1   = np.concatenate([val_leaf_pca,   val_tt_embed],   axis=1).astype(np.float32)
X_test_a1  = np.concatenate([test_leaf_pca,  test_tt_embed],  axis=1).astype(np.float32)
print(f'Fusion input dim: {X_train_a1.shape[1]}')

# ---------- Fusion MLP ----------
class FusionMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes))
    def forward(self, x): return self.net(x)

model = FusionMLP(X_train_a1.shape[1]).to(device)
print(f'FusionMLP params: {sum(p.numel() for p in model.parameters()):,}')

# ---------- Train ----------
class_counts = np.bincount(y_train.astype(int))
cw = torch.tensor(len(y_train) / (n_classes * class_counts), dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=cw)
opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
BS = 4096

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_a1), torch.from_numpy(y_train)), batch_size=BS, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val_a1),   torch.from_numpy(y_val)),   batch_size=BS, shuffle=False)

best_loss, best_state, counter, epochs = float('inf'), None, 0, 100
patience = 10
start = time.time()
for epoch in range(1, epochs + 1):
    model.train(); tr_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        opt.zero_grad(); loss = criterion(model(Xb), yb)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); tr_loss += loss.item() * Xb.size(0)
    sched.step(); tr_loss /= len(y_train)

    model.eval(); vl_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            vl_loss += criterion(model(Xb), yb).item() * Xb.size(0)
    vl_loss /= len(y_val)

    if vl_loss < best_loss:
        best_loss = vl_loss; best_state = model.state_dict().copy(); counter = 0
    else: counter += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d}/{epochs} | Train: {tr_loss:.4f} | Val: {vl_loss:.4f} | {time.time()-start:.0f}s')
    if counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# ---------- Evaluate ----------
model.load_state_dict(best_state)
model.eval()
preds = []
with torch.no_grad():
    for i in range(0, len(X_test_a1), BS):
        Xb = torch.from_numpy(X_test_a1[i:i+BS]).to(device)
        preds.extend(torch.argmax(model(Xb), dim=1).cpu().numpy())

macro = f1_score(y_test, preds, average='macro')
wgt   = f1_score(y_test, preds, average='weighted')
print(f'\n--- Approach 1 (Leaf PCA + TT) Results ---')
print(f'Macro F1:  {macro:.4f}')
print(f'Weighted:  {wgt:.4f}')
print(classification_report(y_test, preds, target_names=class_names, digits=4))

# ---------- Save ----------
torch.save({
    'model_state_dict': best_state,
    'fusion_in_dim': X_train_a1.shape[1],
    'pca_components': pca.components_,
    'pca_mean': pca.mean_,
    'test_macro_f1': float(macro),
    'test_weighted_f1': float(wgt),
}, f'{OUT_DIR}/model_a1_leaf_pca.pth')
print(f'Saved to {OUT_DIR}/model_a1_leaf_pca.pth')
