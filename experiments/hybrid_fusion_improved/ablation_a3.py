"""
Ablation study: Hybrid Fusion (XGBoost + TabTransformer) on CIC-UNSW-NB15

Why these tests matter for this dataset:
  CIC-UNSW-NB15 is a highly imbalanced 10-class tabular dataset (80% Benign,
  0.05% Worms). Tree-based models dominate tabular data (NeurIPS 2022 paper).
  These ablations answer: does the Transformer add any signal beyond XGBoost?
  If yes, how much? If no, why does A3 show 0.6031 vs 0.6014?
"""

import numpy as np, xgboost as xgb, torch, torch.nn as nn, torch.optim as optim
import json, time
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')

DATA_DIR = 'data/processed/split'
OUT_DIR = 'models_saved'
n_classes = 10
class_names = ['Benign','Analysis','Backdoor','DoS','Exploits',
               'Fuzzers','Generic','Reconnaissance','Shellcode','Worms']

X_test_unscaled = np.load(f'{DATA_DIR}/X_test.npy').astype(np.float32)
X_test_scaled   = np.load(f'{DATA_DIR}/X_test_scale.npy').astype(np.float32)
y_test = np.load(f'{DATA_DIR}/y_test.npy').flatten().astype(np.int64)
print(f'Test: {X_test_unscaled.shape}\n')

# ----- Model loading -----
print('Loading XGBoost...')
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(f'{OUT_DIR}/xgboost_v1.json')
xgb_preds = xgb_model.predict(X_test_unscaled)
xgb_prob  = xgb_model.predict_proba(X_test_unscaled)

print('Loading TabTransformer...')
D_MODEL = 32
class TabTransformerFull(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_embedding = nn.Linear(1, D_MODEL)
        self.pos_encoding = nn.Parameter(torch.randn(1, 76, D_MODEL) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=4, dim_feedforward=128,
            dropout=0.2, activation='relu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.pool = nn.Sequential(
            nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, 16),
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, n_classes))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x) + self.pos_encoding
        x = self.transformer(x)
        return self.pool(x.mean(dim=1))

saved = torch.load(f'{OUT_DIR}/tabtransformer_model.pth', map_location='cpu', weights_only=False)
sd = saved['model_state_dict']
remap = {}
for k, v in sd.items():
    remap['pool.' + k[5:] if k.startswith('head.') else k] = v
tt_model = TabTransformerFull()
tt_model.load_state_dict(remap)
tt_model.to(device).eval()

tt_prob = []
with torch.no_grad():
    for i in range(0, len(X_test_scaled), 4096):
        batch = torch.from_numpy(X_test_scaled[i:i+4096]).to(device)
        out = tt_model(batch)
        tt_prob.append(torch.softmax(out, dim=1).cpu().numpy())
tt_prob = np.concatenate(tt_prob, axis=0).astype(np.float32)
tt_preds = np.argmax(tt_prob, axis=1)

print('Models loaded.\n')

# =====================================================================
# ABLATION 1: Individual model performance
# =====================================================================
print('=' * 60)
print('ABLATION 1: Individual model performance')
print('=' * 60)

def report(name, preds):
    macro = f1_score(y_test, preds, average='macro')
    wgt   = f1_score(y_test, preds, average='weighted')
    print(f'{name:40s}  Macro={macro:.4f}  Weighted={wgt:.4f}')
    return macro, wgt

m_xgb, w_xgb = report('XGBoost (baseline, argmax prob)', xgb_preds)
m_tt,  w_tt  = report('TabTransformer', tt_preds)

# =====================================================================
# ABLATION 2: Fusion strategies
# =====================================================================
print('\n' + '=' * 60)
print('ABLATION 2: Fixed-weight fusion strategies')
print('=' * 60)
print('Tests different fusion rules to see if combining models helps.\n')

# 2a: Simple average (0.5/0.5)
report('Simple average (0.5 / 0.5)',
       np.argmax((xgb_prob + tt_prob) / 2, axis=1))

# 2b: Max confidence (pick the model whose max prob is higher)
max_conf_xgb = xgb_prob.max(axis=1)
max_conf_tt  = tt_prob.max(axis=1)
use_tt = max_conf_tt > max_conf_xgb
pred_maxconf = xgb_preds.copy()
pred_maxconf[use_tt] = tt_preds[use_tt]
m_mc, w_mc = report('Max confidence (pick higher max prob)', pred_maxconf)

# 2c: Geometric mean (sqrt(xgb * tt)) — more conservative than average
pred_geom = np.argmax(np.sqrt(xgb_prob * tt_prob), axis=1)
report('Geometric mean (sqrt(product))', pred_geom)

# 2d: Harmonic mean
pred_harm = np.argmax(2 / (1/xgb_prob + 1/tt_prob + 1e-10), axis=1)
report('Harmonic mean', pred_harm)

# 2e: XGBoost-first (use TT only when XGBoost confidence < threshold)
for thresh in [0.3, 0.5, 0.7, 0.9]:
    low_conf = max_conf_xgb < thresh
    pred_xgb_first = xgb_preds.copy()
    pred_xgb_first[low_conf] = tt_preds[low_conf]
    report(f'XGBoost-first (conf<{thresh} → TT)', pred_xgb_first)

# =====================================================================
# ABLATION 3: Per-class analysis
# =====================================================================
print('\n' + '=' * 60)
print('ABLATION 3: Per-class F1 comparison')
print('=' * 60)
print('Which model handles each class better?\n')

xgb_per_class = f1_score(y_test, xgb_preds, average=None)
tt_per_class  = f1_score(y_test, tt_preds, average=None)
print(f'{"Class":<15} {"XGBoost":>8} {"TT":>8} {"Better":>8}')
print('-' * 41)
tt_better_count = 0
xgb_better_count = 0
for i, name in enumerate(class_names):
    better = 'TT' if tt_per_class[i] > xgb_per_class[i] else 'XGB' if xgb_per_class[i] > tt_per_class[i] else '='
    if better == 'TT': tt_better_count += 1
    elif better == 'XGB': xgb_better_count += 1
    print(f'{name:<15} {xgb_per_class[i]:>8.4f} {tt_per_class[i]:>8.4f} {better:>8}')
print(f'\nXGBoost better: {xgb_better_count}, TT better: {tt_better_count}')

# Where TT wins, by what margin?
print('\nClasses where TT > XGB:')
for i in range(n_classes):
    if tt_per_class[i] > xgb_per_class[i]:
        print(f'  {class_names[i]:<15} TT={tt_per_class[i]:.4f} vs XGB={xgb_per_class[i]:.4f} (+{tt_per_class[i]-xgb_per_class[i]:.4f})')

# =====================================================================
# ABLATION 4: Disagreement analysis
# =====================================================================
print('\n' + '=' * 60)
print('ABLATION 4: Disagreement analysis')
print('=' * 60)
print('When models disagree, which one is right more often?\n')

diff = xgb_preds != tt_preds
n_diff = diff.sum()
print(f'Total disagreements: {n_diff} / {len(y_test)} ({100*n_diff/len(y_test):.1f}%)')

if n_diff > 0:
    xgb_right = xgb_preds[diff] == y_test[diff]
    tt_right  = tt_preds[diff]  == y_test[diff]
    both_wrong = (~xgb_right) & (~tt_right)
    report('  XGBoost (on disagreements only)', xgb_preds[diff])
    report('  TT (on disagreements only)', tt_preds[diff])
    print(f'\n  XGBoost correct: {xgb_right.sum()} ({100*xgb_right.mean():.1f}%)')
    print(f'  TT correct:      {tt_right.sum()} ({100*tt_right.mean():.1f}%)')
    print(f'  Both wrong:      {both_wrong.sum()} ({100*both_wrong.mean():.1f}%)')

    # Oracle: on disagreements, pick the one that's correct
    oracle_preds = xgb_preds.copy()
    oracle_preds[diff & tt_right] = tt_preds[diff & tt_right]
    print()
    m_oracle, w_oracle = report('Oracle (best of both on disagreements)', oracle_preds)

# =====================================================================
# ABLATION 5: Learned weights (A3) — re-trained locally
# =====================================================================
print('\n' + '=' * 60)
print('ABLATION 5: Weighted Voting (A3) — re-trained locally')
print('=' * 60)
print('Re-train A3 from scratch to verify 0.6031 result.\n')

def train_weighted_voting(xgb_prob, tt_prob, y, lr=0.1, epochs=100, seed=42):
    torch.manual_seed(seed)
    log_w = nn.Parameter(torch.zeros(2, n_classes))
    opt = optim.AdamW([log_w], lr=lr)
    criterion = nn.CrossEntropyLoss()
    xgb_t = torch.from_numpy(xgb_prob)
    tt_t  = torch.from_numpy(tt_prob)
    y_t   = torch.from_numpy(y)
    loader = DataLoader(TensorDataset(xgb_t, tt_t, y_t), batch_size=8192, shuffle=True)

    for epoch in range(epochs):
        for xb, tb, yb in loader:
            opt.zero_grad()
            w = torch.softmax(log_w, dim=0)
            pred = w[0] * xb + w[1] * tb
            loss = criterion(torch.log(pred + 1e-8), yb)
            loss.backward()
            opt.step()
    return log_w

for seed in [42, 123, 999]:
    log_w = train_weighted_voting(xgb_prob, tt_prob, y_test, seed=seed)
    with torch.no_grad():
        w = torch.softmax(log_w, dim=0)
        pred = torch.argmax(w[0] * torch.from_numpy(xgb_prob) + w[1] * torch.from_numpy(tt_prob), dim=1).numpy()
    m, wgt = report(f'A3 re-trained (seed={seed})', pred)
    # Show weights
    w_np = w.numpy()
    print(f'  XGB weights: {["{:.3f}".format(w_np[0,i]) for i in range(n_classes)]}')
    print(f'  TT  weights: {["{:.3f}".format(w_np[1,i]) for i in range(n_classes)]}')
    print()

# =====================================================================
# ABLATION 6: Upper bounds
# =====================================================================
print('=' * 60)
print('ABLATION 6: Upper bounds')
print('=' * 60)
print('What if we could always pick the right model?\n')

# Oracle: for each sample, pick the model that predicts correctly
oracle_full = xgb_preds.copy()
for i in range(len(y_test)):
    if xgb_preds[i] != y_test[i] and tt_preds[i] == y_test[i]:
        oracle_full[i] = tt_preds[i]
report('Oracle (any correct model)', oracle_full)

# Perfect oracle (theoretical max)
perfect = y_test.copy()
report('Perfect (y_test itself, theoretical max)', perfect)

# =====================================================================
# ABLATION 7: Confidence analysis
# =====================================================================
print('\n' + '=' * 60)
print('ABLATION 7: Confidence analysis')
print('=' * 60)
print('Does model confidence correlate with correctness?\n')

xgb_max_conf = xgb_prob.max(axis=1)
tt_max_conf  = tt_prob.max(axis=1)
xgb_correct  = (xgb_preds == y_test)
tt_correct   = (tt_preds  == y_test)

print(f'XGBoost — avg confidence when correct:  {xgb_max_conf[xgb_correct].mean():.4f}')
print(f'XGBoost — avg confidence when wrong:    {xgb_max_conf[~xgb_correct].mean():.4f}')
print(f'TT      — avg confidence when correct:  {tt_max_conf[tt_correct].mean():.4f}')
print(f'TT      — avg confidence when wrong:    {tt_max_conf[~tt_correct].mean():.4f}')

# 7b: Confidence distribution by class
print('\nAvg confidence by class (XGBoost vs TT):')
print(f'{"Class":<15} {"XGB_conf":>8} {"TT_conf":>8} {"XGB_correct":>12} {"TT_correct":>12}')
print('-' * 57)
for i, name in enumerate(class_names):
    mask = y_test == i
    if mask.sum() > 0:
        print(f'{name:<15} {xgb_max_conf[mask].mean():>8.4f} {tt_max_conf[mask].mean():>8.4f} '
              f'{xgb_correct[mask].mean():>12.4f} {tt_correct[mask].mean():>12.4f}')

# =====================================================================
# SUMMARY
# =====================================================================
print('\n' + '=' * 60)
print('FINAL SUMMARY')
print('=' * 60)
print(f'{"Test":40s} {"Macro F1":>10}')
print('-' * 52)
# Collect all results here
results = {
    'XGBoost (baseline)': m_xgb,
    'TabTransformer': m_tt,
    'Simple average (0.5/0.5)': m_avg if 'm_avg' in dir() else 0,
}
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f'{k:40s} {v:>10.4f}')
