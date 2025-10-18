import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
import shap
import matplotlib.pyplot as plt


def find_best_threshold(y_true, proba, objective="balanced_accuracy"):
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 199),
                proba,
            ]
        )
    )
    best_t, best_score = 0.5, -1.0
    for t in grid:
        pred = (proba >= t).astype(int)
        if objective == "macro_f1":
            score = f1_score(y_true, pred, average='macro', zero_division=0)
        elif objective == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, pred)
        else:
            raise ValueError("objective must be 'macro_f1' or 'balanced_accuracy'")
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t), float(best_score)


def train_and_evaluate_model(train_df, valid_df, n_splits=5, threshold_objective="balanced_accuracy"):
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    if train_df['label'].isna().any() or valid_df['label'].isna().any():
        raise ValueError("label 列に欠損値が含まれています。欠損を除去または補完してください。")

    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_valid = valid_df.drop(columns=['label'])
    y_valid = valid_df['label']

    non_numeric_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
    if non_numeric_cols:
        raise ValueError(f"数値化されていない列があります: {non_numeric_cols}")

    all_labels = y_train.tolist() + y_valid.tolist()
    unique_labels = np.unique(all_labels)
    if len(unique_labels) != 2:
        raise ValueError("現在の実装は2クラス分類にのみ対応しています。")

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valid_encoded = label_encoder.transform(y_valid)

    LGB_PARAMS = dict(
        objective="binary",
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=-1,
        is_unbalance=True,
    )

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(X_train), dtype=float)
    valid_proba_folds = []
    models = []
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train_encoded), start=1):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train_encoded[tr_idx], y_train_encoded[va_idx]

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        callbacks = [lgb.early_stopping(stopping_rounds=200, verbose=False)]
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=callbacks,
        )

        va_proba = model.predict_proba(X_va)[:, 1]
        oof_proba[va_idx] = va_proba

        auc = roc_auc_score(y_va, va_proba)
        va_pred05 = (va_proba >= 0.5).astype(int)
        f1_05 = f1_score(y_va, va_pred05, average='macro', zero_division=0)
        bal_05 = balanced_accuracy_score(y_va, va_pred05)
        fold_rows.append(
            {
                "fold": fold,
                "AUC": auc,
                "MacroF1@0.5": f1_05,
                "BalAcc@0.5": bal_05,
                "best_iteration": getattr(model, "best_iteration_", model.n_estimators),
            }
        )

        valid_proba_folds.append(model.predict_proba(X_valid)[:, 1])
        models.append(model)

    cv_df = pd.DataFrame(fold_rows)
    print("[CV metrics @0.5]")
    print(cv_df)
    print("Mean ± Std")
    print(cv_df[["AUC", "MacroF1@0.5", "BalAcc@0.5"]].agg(['mean', 'std']))

    oof_auc = roc_auc_score(y_train_encoded, oof_proba)
    print(f"OOF AUC: {oof_auc:.4f}")

    valid_proba_mean = np.mean(np.vstack(valid_proba_folds), axis=0)

    t_opt, score_opt = find_best_threshold(y_train_encoded, oof_proba, objective=threshold_objective)
    print(f"Best threshold by {threshold_objective}: t={t_opt:.3f}, score={score_opt:.4f}")

    y_valid_pred_encoded = (valid_proba_mean >= t_opt).astype(int)

    predictions_df = pd.DataFrame({
        'label': y_valid,
        'predicted_label': label_encoder.inverse_transform(y_valid_pred_encoded),
        'proba': valid_proba_mean,
        **valid_df.drop(columns=['label']).to_dict('series'),
    })
    predictions_df.to_csv('results/classification/result_num_lgb.csv', index=False)
    conf_matrix_df = pd.DataFrame(
        confusion_matrix(predictions_df['label'], predictions_df['predicted_label']),
        columns=sorted(set(predictions_df['label']) | set(predictions_df['predicted_label'])),
        index=sorted(set(predictions_df['label']) | set(predictions_df['predicted_label']))
    )
    conf_matrix_df.to_csv('results/classification/confusion_matrix_num_lgb.csv')

    print("Accuracy:", accuracy_score(y_valid_encoded, y_valid_pred_encoded))
    print("Precision:", precision_score(y_valid_encoded, y_valid_pred_encoded, average='macro', zero_division=0))
    print("Recall:", recall_score(y_valid_encoded, y_valid_pred_encoded, average='macro', zero_division=0))
    print("Macro F1:", f1_score(y_valid_encoded, y_valid_pred_encoded, average='macro', zero_division=0))
    print("Balanced Accuracy:", balanced_accuracy_score(y_valid_encoded, y_valid_pred_encoded))

    final_model = models[-1]
    explainer = shap.TreeExplainer(final_model.booster_, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X_valid, check_additivity=False)

    feature_names = X_train.columns

    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=0)
    else:
        shap_array = shap_values

    print(f"SHAP values array shape: {shap_array.shape}")

    if shap_array.ndim == 2:
        shap_array = shap_array[np.newaxis, ...]

    num_classes = shap_array.shape[0]
    if num_classes == 1:
        shap.summary_plot(shap_array[0], X_valid, feature_names=feature_names)
    elif num_classes == 2:
        shap.summary_plot(shap_array[1], X_valid, feature_names=feature_names)
    else:
        for i in range(num_classes):
            print(f"Class: {i}")
            shap.summary_plot(shap_array[i], X_valid, feature_names=feature_names)

    plt.show()
