"""
XGBoost hyperparameter tuning for CIC-UNSW-NB15.
"""
import numpy as np, time, os, json, warnings, functools, random
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import ParameterGrid
import xgboost as xgb

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

print('\n--- Hyperparameter search (20-40 min) ---')
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

print('\n--- Best XGBoost on test ---')
full_report(f'XGBoost tuned (val_macro={best_macro:.4f})', best_xgb, X_test, y_test)

best_xgb.save_model(f'{OUT_DIR}/xgboost_best.json')
with open(f'{OUT_DIR}/xgboost_best_params.json', 'w') as f:
    json.dump({'val_macro': float(best_macro), 'params': best_params}, f, indent=2)
print(f'Saved to {OUT_DIR}/xgboost_best.json')
