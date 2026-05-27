"""
AutoML Pipeline for AF Classification
Entry point: python -m automl_pipeline --data-dir "F:\Automl_test_data\af_class"
"""
import argparse
import json
import os
import sys
import time
import warnings
import pickle
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score
)

warnings.filterwarnings("ignore")

# Try importing tree models
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not installed, will skip LGB models")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed, will skip XGB models")

try:
    import catboost as cb
    HAS_CTB = True
except ImportError:
    HAS_CTB = False
    print("WARNING: catboost not installed, will skip CTB models")

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not installed, will use default params")


# ─── Configuration ───────────────────────────────────────────────
TARGET_COL = "af_annotation_label"
DROP_COLS = {"file_name", "source_record", "dataset_split"}
SEED = 42
N_FOLDS = 5
OPTUNA_N_TRIALS = 150
OPTUNA_TIMEOUT = 1800  # 30 min


# ─── Data Loading ────────────────────────────────────────────────
def load_data(data_dir):
    train_path = os.path.join(data_dir, "anzhen2022_2024_move_cut_features_train.csv")
    test_path = os.path.join(data_dir, "anzhen2022_2024_move_cut_features_test.csv")

    print(f"Loading train from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading test from: {test_path}")
    test_df = pd.read_csv(test_path)

    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Label distribution
    print(f"\n--- Label Distribution (Train) ---")
    print(train_df[TARGET_COL].value_counts().to_string())
    print(f"\n--- Label Distribution (Test) ---")
    print(test_df[TARGET_COL].value_counts().to_string())

    # Missing values
    print(f"\n--- Missing Values (Train) ---")
    missing = train_df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing.to_string())
    else:
        print("No missing values")

    # Data types
    print(f"\n--- Data Types ---")
    print(train_df.dtypes.value_counts().to_string())

    return train_df, test_df


# ─── Feature Engineering ─────────────────────────────────────────
def prepare_features(train_df, test_df):
    """Prepare features, handle preprocessing."""
    # Identify feature columns
    feature_cols = [c for c in train_df.columns
                    if c not in DROP_COLS and c != TARGET_COL]

    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols[:5]}...")

    # Separate target
    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print(f"Classes: {le.classes_}")
    print(f"Encoded classes: {list(range(len(le.classes_)))}")

    # Fill missing with median
    for col in feature_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    # Convert to numeric
    for col in feature_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    # Fill any remaining NaN with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Drop constant columns
    constant_cols = []
    for col in feature_cols:
        if X_train[col].nunique() <= 1:
            constant_cols.append(col)
    if constant_cols:
        print(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
        X_train = X_train.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols)
        feature_cols = [c for c in feature_cols if c not in constant_cols]

    # Drop near-zero variance columns
    nzv_cols = []
    for col in feature_cols:
        vals = X_train[col].values
        if vals.std() < 1e-10:
            nzv_cols.append(col)
    if nzv_cols:
        print(f"Dropping {len(nzv_cols)} near-zero variance columns")
        X_train = X_train.drop(columns=nzv_cols)
        X_test = X_test.drop(columns=nzv_cols)
        feature_cols = [c for c in feature_cols if c not in nzv_cols]

    print(f"\nFinal feature count: {len(feature_cols)}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    return X_train.values, X_test.values, y_train_enc, y_test_enc, le, feature_cols


# ─── Model Definitions ───────────────────────────────────────────
def get_lgb_params(trial=None):
    """LightGBM params with optional Optuna trial."""
    if trial is not None:
        return {
            'num_leaves': trial.suggest_int('num_leaves', 16, 512),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 3000),
            'max_depth': trial.suggest_int('max_depth', -1, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        }
    return {
        'num_leaves': 63,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }


def get_xgb_params(trial=None):
    """XGBoost params with optional Optuna trial."""
    if trial is not None:
        return {
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 3000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'gamma': trial.suggest_float('gamma', 0, 10),
        }
    return {
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'gamma': 0,
    }


def get_ctb_params(trial=None):
    """CatBoost params with optional Optuna trial."""
    if trial is not None:
        return {
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 3000),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 20),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 2),
        }
    return {
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'max_depth': 6,
        'l2_leaf_reg': 3,
        'bagging_temperature': 0.5,
        'random_strength': 1,
    }


# ─── Cross-Validation Evaluator ──────────────────────────────────
def evaluate_model(model_class, params, X, y, n_folds=N_FOLDS, model_name="model"):
    """Stratified K-Fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_scores = []
    fold_preds = np.zeros(len(y), dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = model_class(**params, random_state=SEED)

        # Fit with early stopping if supported
        fit_kwargs = {}
        if 'early_stopping_rounds' in model.__class__().fit.__code__.co_varnames or \
           'early_stopping_rounds' in str(model.fit.__doc__):
            fit_kwargs['early_stopping_rounds'] = 50
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            fit_kwargs['verbose'] = 0

        try:
            model.fit(X_tr, y_tr, **fit_kwargs)
        except TypeError:
            # Some models don't support eval_set
            model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        fold_preds[val_idx] = preds
        score = f1_score(y_val, preds, average='macro')
        fold_scores.append(score)
        print(f"  [{model_name}] Fold {fold_idx+1}: F1={score:.4f}")

    mean_f1 = np.mean(fold_scores)
    print(f"  [{model_name}] Mean F1: {mean_f1:.4f} ± {np.std(fold_scores):.4f}")
    return mean_f1, fold_scores, fold_preds


def evaluate_with_early_stopping(model_class, params, X, y, n_folds=N_FOLDS, model_name="model"):
    """Cross-validation with early stopping for tree models."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_scores = []
    best_n_estimators_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Use a large n_estimators for early stopping, override if present
        fit_params = {**params, 'random_state': SEED}
        fit_params['n_estimators'] = 5000

        model = model_class(**fit_params)

        try:
            v = False if 'CatBoost' in str(model_class) else 0
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=v
            )
            best_n = model.best_iteration_ if hasattr(model, 'best_iteration_') else params.get('n_estimators', 1000)
        except Exception:
            model = model_class(**params, random_state=SEED)
            model.fit(X_tr, y_tr)
            best_n = params.get('n_estimators', 1000)

        best_n_estimators_list.append(best_n)
        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average='macro')
        fold_scores.append(score)
        print(f"  [{model_name}] Fold {fold_idx+1}: F1={score:.4f}, best_iter={best_n}")

    mean_f1 = np.mean(fold_scores)
    avg_best_n = int(np.mean(best_n_estimators_list))
    print(f"  [{model_name}] Mean F1: {mean_f1:.4f} ± {np.std(fold_scores):.4f}, avg_best_n={avg_best_n}")
    return mean_f1, fold_scores, avg_best_n


# ─── Optuna Optimization ─────────────────────────────────────────
def optuna_optimize(model_type, X, y, n_trials, timeout):
    """Optuna hyperparameter search with CV."""
    if not HAS_OPTUNA:
        print("Optuna not available, skipping optimization")
        return None

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )

    def objective(trial):
        if model_type == 'lgb':
            params = get_lgb_params(trial)
            model_class = lgb.LGBMClassifier
        elif model_type == 'xgb':
            params = get_xgb_params(trial)
            model_class = xgb.XGBClassifier
        elif model_type == 'ctb':
            params = get_ctb_params(trial)
            model_class = cb.CatBoostClassifier

        # Quick 3-fold CV for pruning
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # verbose: int for LGB/XGB, bool for CTB
            v = False if model_type == 'ctb' else 0
            model = model_class(**params, random_state=SEED, verbose=v)
            try:
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=30,
                    verbose=v
                )
            except Exception:
                model.fit(X_tr, y_tr)

            preds = model.predict(X_val)
            scores.append(f1_score(y_val, preds, average='macro'))

        mean_score = np.mean(scores)
        trial.report(mean_score, 0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return mean_score

    print(f"\n{'='*60}")
    print(f"Optuna optimization for {model_type} ({n_trials} trials, {timeout}s timeout)")
    print(f"{'='*60}")

    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    except Exception as e:
        print(f"Optuna optimization interrupted: {e}")

    print(f"\nBest trial for {model_type}:")
    print(f"  Value: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    return study


# ─── Final Training & Evaluation ─────────────────────────────────
def train_final_model(model_class, params, X_train, y_train, X_test, y_test, model_name):
    """Train on full train set and evaluate on test set."""
    print(f"\n{'='*60}")
    print(f"Final training: {model_name}")
    print(f"{'='*60}")

    v = False if model_name == 'CatBoost' else 0
    model = model_class(**params, random_state=SEED, n_estimators=5000, verbose=v)

    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=v
        )
        best_n = model.best_iteration_ if hasattr(model, 'best_iteration_') else params.get('n_estimators', 1000)
        print(f"Best iteration: {best_n}")
    except Exception:
        model = model_class(**params, random_state=SEED, verbose=v)
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average='macro')
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro')
    rec = recall_score(y_test, preds, average='macro')

    print(f"\n--- {model_name} Test Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, preds))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    return model, f1, acc, prec, rec, preds


# ─── Ensemble ─────────────────────────────────────────────────────
def ensemble_predict(models, X):
    """Soft voting ensemble."""
    from collections import Counter
    n_classes = max(max(m.classes_) for m in models) + 1
    all_probs = []
    for model in models:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            # Ensure consistent shape
            if probs.shape[1] < n_classes:
                full_probs = np.zeros((probs.shape[0], n_classes))
                for i, c in enumerate(model.classes_):
                    full_probs[:, c] = probs[:, i]
                probs = full_probs
            all_probs.append(probs)
        else:
            # Hard voting fallback
            preds = model.predict(X)
            probs = np.zeros((len(preds), n_classes))
            for i, p in enumerate(preds):
                probs[i, p] = 1.0
            all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    return np.argmax(avg_probs, axis=1)


# ─── Main Pipeline ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=os.environ.get('LOCAL_DATA_DIR', r'F:\Automl_test_data\af_class'))
    parser.add_argument('--phase', default='full', choices=['inspect', 'baseline', 'optimize', 'full'])
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  AF Classification AutoML Pipeline")
    print("=" * 70)
    print(f"Data dir: {data_dir}")
    print(f"Phase: {args.phase}")
    print(f"Output dir: {output_dir}")
    print(f"LGB: {HAS_LGB}, XGB: {HAS_XGB}, CTB: {HAS_CTB}, Optuna: {HAS_OPTUNA}")
    print()

    t_start = time.time()

    # Phase 1: Load & inspect
    print("\n" + "=" * 70)
    print("  PHASE 1: Data Loading & Inspection")
    print("=" * 70)
    train_df, test_df = load_data(data_dir)

    # Phase 2: Feature preparation
    print("\n" + "=" * 70)
    print("  PHASE 2: Feature Preparation")
    print("=" * 70)
    X_train, X_test, y_train, y_test, le, feature_cols = prepare_features(train_df, test_df)

    if args.phase == 'inspect':
        print("\nInspection complete.")
        return

    # Phase 3: Baseline models
    print("\n" + "=" * 70)
    print("  PHASE 3: Baseline Models (5-Fold CV)")
    print("=" * 70)

    baseline_results = {}

    if HAS_LGB:
        lgb_params = get_lgb_params()
        f1, scores, _ = evaluate_with_early_stopping(
            lgb.LGBMClassifier, lgb_params, X_train, y_train, model_name="LGB_default"
        )
        baseline_results['lgb_default'] = f1

    if HAS_XGB:
        xgb_params = get_xgb_params()
        f1, scores, _ = evaluate_with_early_stopping(
            xgb.XGBClassifier, xgb_params, X_train, y_train, model_name="XGB_default"
        )
        baseline_results['xgb_default'] = f1

    if HAS_CTB:
        ctb_params = get_ctb_params()
        f1, scores, _ = evaluate_with_early_stopping(
            cb.CatBoostClassifier, ctb_params, X_train, y_train, model_name="CTB_default"
        )
        baseline_results['ctb_default'] = f1

    print(f"\nBaseline Results:")
    for name, f1 in sorted(baseline_results.items(), key=lambda x: -x[1]):
        print(f"  {name}: F1={f1:.4f}")

    if args.phase == 'baseline':
        print("\nBaseline complete.")
        return

    # Phase 4: Optuna optimization
    print("\n" + "=" * 70)
    print("  PHASE 4: Hyperparameter Optimization (Optuna)")
    print("=" * 70)

    best_studies = {}
    models_to_optimize = []
    if HAS_LGB:
        models_to_optimize.append(('lgb', lgb.LGBMClassifier, get_lgb_params))
    if HAS_XGB:
        models_to_optimize.append(('xgb', xgb.XGBClassifier, get_xgb_params))
    if HAS_CTB:
        models_to_optimize.append(('ctb', cb.CatBoostClassifier, get_ctb_params))

    for model_type, model_class, param_fn in models_to_optimize:
        study = optuna_optimize(model_type, X_train, y_train, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT)
        if study is not None:
            best_studies[model_type] = study

    # Phase 5: Final training with best params
    print("\n" + "=" * 70)
    print("  PHASE 5: Final Training & Test Evaluation")
    print("=" * 70)

    final_models = {}
    final_results = {}

    for model_type, model_class, param_fn in models_to_optimize:
        if model_type in best_studies:
            best_params = best_studies[model_type].best_trial.params
        else:
            best_params = param_fn()

        # Rebuild params dict
        if model_type == 'lgb':
            params = get_lgb_params(None)
            params.update(best_params)
            params['verbose'] = -1
            params['silent'] = True
        elif model_type == 'xgb':
            params = get_xgb_params(None)
            params.update(best_params)
            params['verbosity'] = 0
        elif model_type == 'ctb':
            params = get_ctb_params(None)
            params.update(best_params)
            params['verbose'] = 0

        model, f1, acc, prec, rec, preds = train_final_model(
            model_class, params, X_train, y_train, X_test, y_test,
            f"{model_type.upper()}_tuned"
        )
        final_models[model_type] = (model_class, params, model)
        final_results[f"{model_type}_tuned"] = {'f1': f1, 'acc': acc, 'prec': prec, 'rec': rec}

    # Phase 6: Ensemble
    print("\n" + "=" * 70)
    print("  PHASE 6: Ensemble (Soft Voting)")
    print("=" * 70)

    if len(final_models) >= 2:
        trained_models = [m for _, (_, _, m) in final_models.items()]
        ens_preds = ensemble_predict(trained_models, X_test)
        ens_f1 = f1_score(y_test, ens_preds, average='macro')
        ens_acc = accuracy_score(y_test, ens_preds)
        ens_prec = precision_score(y_test, ens_preds, average='macro')
        ens_rec = recall_score(y_test, ens_preds, average='macro')

        print(f"\nEnsemble Results:")
        print(f"  Accuracy:  {ens_acc:.4f}")
        print(f"  F1 (macro): {ens_f1:.4f}")
        print(f"  Precision: {ens_prec:.4f}")
        print(f"  Recall:    {ens_rec:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, ens_preds))
        print(f"Confusion Matrix:")
        print(confusion_matrix(y_test, ens_preds))

        final_results['ensemble'] = {'f1': ens_f1, 'acc': ens_acc, 'prec': ens_prec, 'rec': ens_rec}

    # Phase 7: Summary
    t_elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"Total time: {t_elapsed:.1f}s ({t_elapsed/60:.1f}min)")
    print()

    # Find best model
    best_name = max(final_results, key=lambda k: final_results[k]['f1'])
    best_f1 = final_results[best_name]['f1']
    target_met = best_f1 >= 0.95

    print(f"Best model: {best_name}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Target F1 ≥ 0.95: {'✅ MET' if target_met else '❌ NOT MET'}")
    print()

    for name, metrics in sorted(final_results.items(), key=lambda x: -x[1]['f1']):
        marker = " ← BEST" if name == best_name else ""
        print(f"  {name:20s} F1={metrics['f1']:.4f}  Acc={metrics['acc']:.4f}  Prec={metrics['prec']:.4f}  Rec={metrics['rec']:.4f}{marker}")

    # Save outputs
    metrics_output = {
        'best_model': best_name,
        'best_f1': best_f1,
        'target_met': target_met,
        'elapsed_seconds': t_elapsed,
        'all_results': final_results,
        'baseline_results': baseline_results,
        'n_features': len(feature_cols),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'classes': le.classes_.tolist(),
    }

    if best_studies:
        metrics_output['optuna_best'] = {}
        for mt, study in best_studies.items():
            metrics_output['optuna_best'][mt] = {
                'value': study.best_trial.value,
                'params': study.best_trial.params,
                'n_trials': len(study.trials),
            }

    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2, default=str)
    print(f"\nMetrics saved to: {metrics_path}")

    # Save best model
    if best_name != 'ensemble' and best_name.replace('_tuned', '') in final_models:
        model_type_key = best_name.replace('_tuned', '')
        _, _, best_model_obj = final_models[model_type_key]
        model_path = os.path.join(output_dir, 'best_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best_model_obj, f)
        print(f"Best model saved to: {model_path}")

    # Save label encoder
    le_path = os.path.join(output_dir, 'label_encoder.pkl')
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"Label encoder saved to: {le_path}")

    # Save feature columns
    feat_path = os.path.join(output_dir, 'feature_columns.json')
    with open(feat_path, 'w') as f:
        json.dump(feature_cols, f)
    print(f"Feature columns saved to: {feat_path}")

    print(f"\n{'='*70}")
    print(f"Pipeline complete! Best F1={best_f1:.4f} ({'✅' if target_met else '❌'})")
    print(f"{'='*70}")

    return 0 if target_met else 1


if __name__ == '__main__':
    sys.exit(main())
