"""
CatBoost tuning + Feature engineering for CIC-UNSW-NB15.
Run after XGBoost tuning (needs xgboost_best.json for section 4).
"""
import numpy as np, time, os, warnings, functools, random
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import ParameterGrid

print = functools.partial(print, flush=True)
DATA_DIR = 'data/processed/split'
OUT_DIR = 'models_saved'
os.makedirs(OUT_DIR, exist_ok=True)
n_classes = 10
class_names = ['Benign','Analysis','Backdoor','DoS','Exploits',
               'Fuzzers','Generic','Reconnaissance','Shellcode','Worms']

X_train = np.load(f'{DATA_DIR}/X_train.npy').astype(np.float32)
X_val   = np.load(f'{DATA_DIR}/X_val.npy').astype(np.float32)
X_test  = np.load(f'{DATA_DIR}/X_test.npy').astype(np.float32)
y_train = np.load(f'{DATA_DIR}/y_train.npy').flatten().astype(np.int64)
y_val   = np.load(f'{DATA_DIR}/y_val.npy').flatten().astype(np.int64)
y_test  = np.load(f'{DATA_DIR}/y_test.npy').flatten().astype(np.int64)

class_counts = np.bincount(y_train.astype(int))
n_train = len(y_train)
sample_weights = n_train / (n_classes * class_counts)
sw_train = sample_weights[y_train].astype(np.float32)
print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')

results = []

def evaluate(name, model, X_test_, y_test_):
    preds = model.predict(X_test_)
    if preds.ndim == 2 and preds.shape[1] > 1:
        preds = np.argmax(preds, axis=1)
    macro = f1_score(y_test_, preds, average='macro')
    wgt   = f1_score(y_test_, preds, average='weighted')
    print(f'  {name:45s} Macro={macro:.4f}  Weighted={wgt:.4f}')
    results.append({'model': name, 'macro_f1': macro, 'weighted_f1': wgt})
    return macro, wgt, preds

def full_report(name, model, X_test_, y_test_):
    macro, wgt, preds = evaluate(name, model, X_test_, y_test_)
    print()
    print(classification_report(y_test_, preds, target_names=class_names, digits=4))
    return macro, wgt

# =====================================================================
# CatBoost
# =====================================================================
print('\n' + '=' * 70)
print('CatBoost')
print('=' * 70)

from catboost import CatBoostClassifier

print('\n--- CatBoost baseline ---')
model_cat = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.1,
    loss_function='MultiClass', eval_metric='MultiClass',
    bootstrap_type='Bernoulli', subsample=0.8,
    random_seed=42, thread_count=-1, verbose=False,
    early_stopping_rounds=20,
)
model_cat.fit(X_train, y_train, sample_weight=sw_train,
              eval_set=(X_val, y_val), verbose=False)
full_report('CatBoost baseline', model_cat, X_test, y_test)

print('\n--- CatBoost hyperparameter search ---')
cat_param_grid = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bylevel': [0.6, 0.8, 1.0],
    'l2_leaf_reg': [1, 3, 5],
    'min_data_in_leaf': [1, 5, 20],
}
all_cat_params = list(ParameterGrid(cat_param_grid))
random.shuffle(all_cat_params)
search_cat = all_cat_params[:15]

best_cat, best_cat_macro = None, 0.0
for i, p in enumerate(search_cat):
    t0 = time.time()
    model = CatBoostClassifier(
        iterations=500, loss_function='MultiClass', eval_metric='MultiClass',
        bootstrap_type='Bernoulli',
        random_seed=42, thread_count=-1, verbose=False,
        early_stopping_rounds=20, **p,
    )
    model.fit(X_train, y_train, sample_weight=sw_train,
              eval_set=(X_val, y_val), verbose=False)
    preds = model.predict(X_val)
    macro = f1_score(y_val, preds, average='macro')
    print(f'  [{i+1:2d}/{len(search_cat)}] {time.time()-t0:4.0f}s  '
          f'depth={p["depth"]} lr={p["learning_rate"]}  '
          f'→ val_macro={macro:.4f}')
    if macro > best_cat_macro:
        best_cat_macro = macro
        best_cat = model
        best_cat_params = p

print(f'\nBest CatBoost val Macro F1: {best_cat_macro:.4f}')
full_report(f'CatBoost tuned (val_macro={best_cat_macro:.4f})', best_cat, X_test, y_test)
best_cat.save_model(f'{OUT_DIR}/catboost_best.cbm')

# =====================================================================
# Feature engineering
# =====================================================================
print('\n' + '=' * 70)
print('Feature engineering — top interactions')
print('=' * 70)

import xgboost as xgb
best_xgb_model = xgb.XGBClassifier()
best_xgb_model.load_model(f'{OUT_DIR}/xgboost_best.json')
importances = best_xgb_model.feature_importances_
top_k = np.argsort(importances)[-10:]
n_interact = len(top_k)
print(f'Top-10 feature indices: {top_k.tolist()}')

def add_interactions(X):
    X_eng = np.ascontiguousarray(X.copy())
    for i in range(n_interact):
        for j in range(i+1, n_interact):
            col = X[:, top_k[i]] * X[:, top_k[j]]
            X_eng = np.column_stack([X_eng, col.astype(np.float32)])
    return X_eng

n_new = n_interact * (n_interact - 1) // 2
print(f'Adding {n_new} interaction features...')
X_train_eng = add_interactions(X_train)
X_val_eng   = add_interactions(X_val)
X_test_eng  = add_interactions(X_test)
print(f'New dims — Train: {X_train_eng.shape}')

model_eng = xgb.XGBClassifier(
    objective='multi:softprob', num_class=n_classes,
    n_estimators=500, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, eval_metric='mlogloss',
    early_stopping_rounds=20,
)
model_eng.fit(X_train_eng, y_train, sample_weight=sw_train,
              eval_set=[(X_val_eng, y_val)], verbose=False)
full_report('XGBoost + interactions (top 10 feat)', model_eng, X_test_eng, y_test)

# =====================================================================
print('\n' + '=' * 70)
print('FINAL COMPARISON')
print('=' * 70)
print(f'{"Model":45s} {"Macro F1":>10} {"Weighted":>10}')
print('-' * 67)
for r in sorted(results, key=lambda x: -x['macro_f1']):
    print(f'{r["model"]:45s} {r["macro_f1"]:>10.4f} {r["weighted_f1"]:>10.4f}')

with open(f'{OUT_DIR}/tuning_results_catboost_feat.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to {OUT_DIR}/tuning_results_catboost_feat.json')
