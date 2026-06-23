import numpy as np, xgboost as xgb, torch, torch.nn as nn, torch.optim as optim, time
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')

DATA_DIR = 'data/processed/split'
OUT_DIR = 'models_saved'
n_classes = 10
class_names = ['Benign','Analysis','Backdoor','DoS','Exploits',
               'Fuzzers','Generic','Reconnaissance','Shellcode','Worms']

# Load data
X_test_unscaled = np.load(f'{DATA_DIR}/X_test.npy').astype(np.float32)
X_test_scaled   = np.load(f'{DATA_DIR}/X_test_scale.npy').astype(np.float32)
y_test = np.load(f'{DATA_DIR}/y_test.npy').flatten().astype(np.int64)
print(f'Test: {X_test_unscaled.shape}')

# ---------- 1. XGBoost baseline ----------
print('\n=== 1. XGBoost baseline ===')
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(f'{OUT_DIR}/xgboost_v1.json')

xgb_preds = xgb_model.predict(X_test_unscaled)
xgb_prob  = xgb_model.predict_proba(X_test_unscaled)
xgb_macro = f1_score(y_test, xgb_preds, average='macro')
xgb_wgt   = f1_score(y_test, xgb_preds, average='weighted')
print(f'XGBoost baseline: Macro F1={xgb_macro:.4f}, Weighted={xgb_wgt:.4f}')

# ---------- 2. Load TT ----------
print('\n=== 2. TabTransformer probs ===')
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
remap = {}
for k, v in sd.items():
    if k.startswith('head.'): remap['pool.' + k[5:]] = v
    else: remap[k] = v
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

# ---------- 3. Ablation: A3 with fixed weights ----------
print('\n=== 3. Ablation comparisons ===')

# 3a: Pure XGBoost (via A3 with weights [1,0] fixed)
pred_xgb_only = np.argmax(xgb_prob, axis=1)
m_xgb = f1_score(y_test, pred_xgb_only, average='macro')
w_xgb = f1_score(y_test, pred_xgb_only, average='weighted')
print(f'XGBoost (argmax prob):   Macro={m_xgb:.4f}, Wgt={w_xgb:.4f}')

# 3b: Pure TT
pred_tt_only = np.argmax(tt_prob, axis=1)
m_tt = f1_score(y_test, pred_tt_only, average='macro')
w_tt = f1_score(y_test, pred_tt_only, average='weighted')
print(f'TabTransformer (argmax): Macro={m_tt:.4f}, Wgt={w_tt:.4f}')

# 3c: Simple average (0.5/0.5)
pred_avg = np.argmax((xgb_prob + tt_prob) / 2, axis=1)
m_avg = f1_score(y_test, pred_avg, average='macro')
w_avg = f1_score(y_test, pred_avg, average='weighted')
print(f'Simple average (0.5/0.5): Macro={m_avg:.4f}, Wgt={w_avg:.4f}')

# 3d: XGBoost argmax vs TT argmax disagreement
diff_mask = pred_xgb_only != pred_tt_only
n_diff = diff_mask.sum()
print(f'\nDisagreement: {n_diff}/{len(y_test)} ({100*n_diff/len(y_test):.1f}%)')
if n_diff > 0:
    # Where they disagree, which one is correct more often?
    xgb_correct = pred_xgb_only[diff_mask] == y_test[diff_mask]
    tt_correct  = pred_tt_only[diff_mask]  == y_test[diff_mask]
    print(f'  XGBoost correct: {xgb_correct.sum()} ({100*xgb_correct.mean():.1f}%)')
    print(f'  TT correct:      {tt_correct.sum()} ({100*tt_correct.mean():.1f}%)')
    print(f'  Both wrong:      {(~xgb_correct & ~tt_correct).sum()}')

# 3e: Oracle (if either model is correct, pick it)
oracle_mask = np.zeros(len(y_test), dtype=bool)
oracle_mask[diff_mask] = xgb_correct | tt_correct
print(f'\nOracle upper bound (any model correct): {oracle_mask.sum()}/{n_diff} on disagreements')
# Approximate oracle F1: wherever XGBoost is correct or TT is correct
oracle_preds = pred_xgb_only.copy()
oracle_preds[diff_mask & tt_correct] = pred_tt_only[diff_mask & tt_correct]
m_oracle = f1_score(y_test, oracle_preds, average='macro')
w_oracle = f1_score(y_test, oracle_preds, average='weighted')
print(f'Oracle (best of both):  Macro={m_oracle:.4f}, Wgt={w_oracle:.4f}')

# 3f: A3 learned weights from saved model
print('\n=== 4. A3 with saved learned weights ===')
import json
with open(f'{OUT_DIR}/model_a3_learned_weights.json') as f:
    w = json.load(f)
# Weights are ~1.0 for XGBoost per the earlier output
xgb_w_vals = np.array([w[c]['XGBoost_Weight'] for c in class_names], dtype=np.float32)
tt_w_vals  = np.array([w[c]['Transformer_Weight'] for c in class_names], dtype=np.float32)

# Weighted voting with these weights
a3_preds = np.argmax(xgb_prob * xgb_w_vals + tt_prob * tt_w_vals, axis=1)
m_a3 = f1_score(y_test, a3_preds, average='macro')
w_a3 = f1_score(y_test, a3_preds, average='weighted')
print(f'A3 (saved weights):      Macro={m_a3:.4f}, Wgt={w_a3:.4f}')

# 3g: Compare A3 vs XGBoost predictions
a3_diff = (a3_preds != pred_xgb_only).sum()
print(f'A3 differs from XGBoost: {a3_diff} samples ({100*a3_diff/len(y_test):.2f}%)')

# Summary
print('\n' + '=' * 50)
print('SUMMARY')
print('=' * 50)
print(f'XGBoost (argmax prob):   {m_xgb:.4f} / {w_xgb:.4f}')
print(f'TabTransformer:          {m_tt:.4f} / {w_tt:.4f}')
print(f'Simple average:          {m_avg:.4f} / {w_avg:.4f}')
print(f'A3 (saved weights):      {m_a3:.4f} / {w_a3:.4f}')
print(f'Oracle upper bound:      {m_oracle:.4f} / {w_oracle:.4f}')
