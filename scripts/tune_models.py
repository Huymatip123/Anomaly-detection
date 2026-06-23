"""
Comprehensive model tuning: XGBoost, LightGBM, CatBoost + feature engineering
for CIC-UNSW-NB15 (76 features, 10 classes, imbalanced).

Runs in sections — comment out what you don't need.
"""

import numpy as np, time, os, json, warnings, functools
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

# Balanced sample weights (used by all tree models)
class_counts = np.bincount(y_train.astype(int))
n_train = len(y_train)
sample_weights = n_train / (n_classes * class_counts)
sw_train = sample_weights[y_train].astype(np.float32)
print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
print(f'Sample weights: {sample_weights}')

results = []

def evaluate(name, model, X_test_, y_test_):
    preds = model.predict(X_test_)
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
# SECTION 1: XGBoost hyperparameter tuning
# =====================================================================
print('\n' + '=' * 70)
print('SECTION 1: XGBoost hyperparameter tuning')
print('=' * 70)

import xgboost as xgb

# Baseline (original params)
print('\n--- Baseline re-fit ---')
model_base = xgb.XGBClassifier(
    objective='multi:softprob', num_class=n_classes,
    n_estimators=500, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, eval_metric='mlogloss',
    early_stopping_rounds=20,
)
model_base.fit(X_train, y_train, sample_weight=sw_train,
               eval_set=[(X_val, y_val)], verbose=False)
full_report('XGBoost baseline (re-fit)', model_base, X_test, y_test)

# Tuning search
print('\n--- Hyperparameter search (may take 20-40 min) ---')
param_grid = {
    'n_estimators': [1000],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1],
    'reg_alpha': [0, 0.1, 1.0],
}
# Limit to 24 random combinations to keep runtime reasonable
import random
random.seed(42)
all_params = list(ParameterGrid(param_grid))
random.shuffle(all_params)
search_params = all_params[:24]
print(f'Testing {len(search_params)} configurations...')

best_xgb, best_macro = None, 0.0
for i, params in enumerate(search_params):
    t0 = time.time()
    model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=n_classes,
        random_state=42, n_jobs=-1, eval_metric='mlogloss',
        early_stopping_rounds=20, **params,
    )
    model.fit(X_train, y_train, sample_weight=sw_train,
              eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    macro = f1_score(y_val, preds, average='macro')
    elapsed = time.time() - t0
    print(f'  [{i+1:2d}/{len(search_params)}] {elapsed:4.0f}s  depth={params["max_depth"]} '
          f'lr={params["learning_rate"]} sub={params["subsample"]} col={params["colsample_bytree"]} '
          f'mcw={params["min_child_weight"]} gamma={params["gamma"]} '
          f'alpha={params["reg_alpha"]}  → val_macro={macro:.4f}')
    if macro > best_macro:
        best_macro = macro
        best_xgb = model
        best_params = params

print(f'\nBest val Macro F1: {best_macro:.4f}')
print(f'Best params: {best_params}')

# Evaluate best XGBoost on test
print('\n--- Best XGBoost on test ---')
full_report(f'XGBoost tuned (val_macro={best_macro:.4f})', best_xgb, X_test, y_test)

# Save
best_xgb.save_model(f'{OUT_DIR}/xgboost_best.json')
with open(f'{OUT_DIR}/xgboost_best_params.json', 'w') as f:
    json.dump({'val_macro': float(best_macro), 'params': best_params}, f, indent=2)
print(f'Saved to {OUT_DIR}/xgboost_best.json')


# =====================================================================
# SECTION 2: LightGBM (requires `pip install lightgbm`)
# =====================================================================
print('\n' + '=' * 70)
print('SECTION 2: LightGBM')
print('=' * 70)

try:
    import lightgbm as lgb

    # Baseline LightGBM
    print('\n--- LightGBM baseline ---')
    lgb_train = lgb.Dataset(X_train, y_train, weight=sw_train)
    lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train)

    params_lgb = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'num_threads': -1,
        'seed': 42,
    }
    model_lgb = lgb.train(params_lgb, lgb_train,
                          valid_sets=[lgb_val], num_boost_round=500,
                          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    preds_lgb = np.argmax(model_lgb.predict(X_test), axis=1)
    full_report('LightGBM baseline', model_lgb, X_test, y_test)

    # LightGBM tuning
    print('\n--- LightGBM hyperparameter search ---')
    lgb_param_grid = {
        'num_leaves': [15, 31, 63, 127],
        'max_depth': [-1, 8, 12],
        'learning_rate': [0.01, 0.05, 0.1],
        'feature_fraction': [0.6, 0.8, 1.0],
        'bagging_fraction': [0.6, 0.8, 1.0],
        'min_child_samples': [5, 20, 50],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [0, 0.1, 1.0],
    }
    all_lgb_params = list(ParameterGrid(lgb_param_grid))
    random.shuffle(all_lgb_params)
    search_lgb = all_lgb_params[:20]

    best_lgb, best_lgb_macro = None, 0.0
    for i, p in enumerate(search_lgb):
        t0 = time.time()
        params_lgb_tune = {
            'objective': 'multiclass', 'num_class': n_classes,
            'metric': 'multi_logloss', 'boosting_type': 'gbdt',
            'bagging_freq': 5, 'verbose': -1, 'num_threads': -1, 'seed': 42,
            **p,
        }
        model = lgb.train(params_lgb_tune, lgb_train,
                          valid_sets=[lgb_val], num_boost_round=500,
                          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        preds = np.argmax(model.predict(X_val), axis=1)
        macro = f1_score(y_val, preds, average='macro')
        print(f'  [{i+1:2d}/{len(search_lgb)}] {time.time()-t0:4.0f}s  '
              f'leaves={p["num_leaves"]} lr={p["learning_rate"]} '
              f'→ val_macro={macro:.4f}')
        if macro > best_lgb_macro:
            best_lgb_macro = macro
            best_lgb = model
            best_lgb_params = p

    print(f'\nBest LightGBM val Macro F1: {best_lgb_macro:.4f}')
    full_report(f'LightGBM tuned (val_macro={best_lgb_macro:.4f})', best_lgb, X_test, y_test)
    best_lgb.save_model(f'{OUT_DIR}/lightgbm_best.txt')

except ImportError:
    print('LightGBM not installed. Install with: pip install lightgbm')


# =====================================================================
# SECTION 3: CatBoost (requires `pip install catboost`)
# =====================================================================
print('\n' + '=' * 70)
print('SECTION 3: CatBoost')
print('=' * 70)

try:
    from catboost import CatBoostClassifier

    # Baseline CatBoost
    print('\n--- CatBoost baseline ---')
    model_cat = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.1,
        loss_function='MultiClass', eval_metric='MultiClass',
        random_seed=42, thread_count=-1, verbose=False,
        early_stopping_rounds=20,
    )
    model_cat.fit(X_train, y_train, sample_weight=sw_train,
                  eval_set=(X_val, y_val), verbose=False)
    full_report('CatBoost baseline', model_cat, X_test, y_test)

    # CatBoost tuning
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
            random_seed=42, thread_count=-1, verbose=False,
            early_stopping_rounds=20, **p,
        )
        model.fit(X_train, y_train, sample_weight=sw_train,
                  eval_set=(X_val, y_val), verbose=False)
        preds = model.predict(X_val)
        macro = f1_score(y_val, preds, average='macro')
        print(f'  [{i+1:2d}/{len(search_cat)}] {time.time()-t0:4.0f}s  '
              f'depth={p["depth"]} lr={p["learning_rate"]} '
              f'→ val_macro={macro:.4f}')
        if macro > best_cat_macro:
            best_cat_macro = macro
            best_cat = model
            best_cat_params = p

    print(f'\nBest CatBoost val Macro F1: {best_cat_macro:.4f}')
    full_report(f'CatBoost tuned (val_macro={best_cat_macro:.4f})', best_cat, X_test, y_test)
    best_cat.save_model(f'{OUT_DIR}/catboost_best.cbm')

except ImportError:
    print('CatBoost not installed. Install with: pip install catboost')


# =====================================================================
# SECTION 4: Feature engineering (interaction features)
# =====================================================================
print('\n' + '=' * 70)
print('SECTION 4: Feature engineering — top interactions')
print('=' * 70)
print('Adding pairwise interactions between top-10 most important XGBoost features.\n')

best_xgb_model = xgb.XGBClassifier()
best_xgb_model.load_model(f'{OUT_DIR}/xgboost_best.json')
importances = best_xgb_model.feature_importances_
top_k = np.argsort(importances)[-10:]  # indices of top 10 features
n_interact = len(top_k)
print(f'Top-10 feature indices: {top_k.tolist()}')

# Create interaction features: multiply each pair of top features
def add_interactions(X):
    X_eng = X.copy()
    X_eng = np.ascontiguousarray(X_eng)
    for i in range(n_interact):
        for j in range(i+1, n_interact):
            col = X[:, top_k[i]] * X[:, top_k[j]]
            X_eng = np.column_stack([X_eng, col.astype(np.float32)])
    return X_eng

print(f'Adding {n_interact * (n_interact-1) // 2} interaction features...')
X_train_eng = add_interactions(X_train)
X_val_eng   = add_interactions(X_val)
X_test_eng  = add_interactions(X_test)
print(f'New dims — Train: {X_train_eng.shape}, Val: {X_val_eng.shape}, Test: {X_test_eng.shape}')

model_eng = xgb.XGBClassifier(
    objective='multi:softprob', num_class=n_classes,
    n_estimators=500, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, eval_metric='mlogloss',
    early_stopping_rounds=20,
)
model_eng.fit(X_train_eng, y_train, sample_weight=sw_train,
              eval_set=[(X_val_eng, y_val)], verbose=False)
full_report('XGBoost + interactions (top 10 features)', model_eng, X_test_eng, y_test)


# =====================================================================
# FINAL COMPARISON
# =====================================================================
print('\n' + '=' * 70)
print('FINAL COMPARISON')
print('=' * 70)
print(f'{"Model":45s} {"Macro F1":>10} {"Weighted":>10}')
print('-' * 67)
for r in sorted(results, key=lambda x: -x['macro_f1']):
    print(f'{r["model"]:45s} {r["macro_f1"]:>10.4f} {r["weighted_f1"]:>10.4f}')

# Save results
with open(f'{OUT_DIR}/tuning_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to {OUT_DIR}/tuning_results.json')
