import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
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
import os


def find_best_threshold(y_true, proba, objective="balanced_accuracy", pos_label=1, beta=1.0):
    """Find an optimal probability threshold.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0/1).
    proba : array-like of shape (n_samples,)
        Predicted probabilities for class 1.
    objective : str
        One of:
        - "f1_pos": F1 for the positive class (pos_label)
        - "fbeta_pos": F-beta for the positive class (pos_label)
        - "macro_f1": Macro-averaged F1 over classes 0 and 1
        - "balanced_accuracy": Balanced accuracy
    pos_label : int
        Which label is treated as the positive class (default=1).
    beta : float
        Beta for F-beta (only used when objective="fbeta_pos").
    """
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 199),
                np.asarray(proba, dtype=float),
            ]
        )
    )

    best_t, best_score = 0.5, -1.0
    for t in grid:
        pred = (proba >= t).astype(int)

        if objective == "f1_pos":
            score = f1_score(y_true, pred, average="binary", pos_label=pos_label, zero_division=0)
        elif objective == "fbeta_pos":
            from sklearn.metrics import fbeta_score
            score = fbeta_score(y_true, pred, beta=beta, pos_label=pos_label, zero_division=0)
        elif objective == "macro_f1":
            score = f1_score(y_true, pred, average="macro", zero_division=0)
        elif objective == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, pred)
        else:
            raise ValueError(
                "objective must be one of 'f1_pos', 'fbeta_pos', 'macro_f1', or 'balanced_accuracy'"
            )

        if score > best_score:
            best_score, best_t = score, t

    return float(best_t), float(best_score)


def train_and_evaluate_model(
    train_df,
    valid_df,
    n_splits=5,
    threshold_objective="balanced_accuracy",
    create_shap_dependence=False,
    dependence_features=None,
    max_dependence=5,
    save_shap_plots=True,
    shap_plot_dir="results/classification",
):
    train_df = train_df.copy()
    valid_df = valid_df.copy()


    if train_df['label'].isna().any() or valid_df['label'].isna().any():
        raise ValueError("label 列に欠損値が含まれています。欠損を除去または補完してください。")

    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_valid = valid_df.drop(columns=['label'])
    y_valid = valid_df['label']

    neg, pos = np.bincount(y_train)  # XGBoost 用の scale_pos_weight
    scale_pos_weight = neg / pos

    if 'sentence' not in X_train.columns:
        raise ValueError("'sentence' 列がデータに含まれていません。")

    train_sentence = X_train.pop('sentence').fillna('').astype(str)
    valid_sentence = X_valid.pop('sentence').fillna('').astype(str)

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
        #is_unbalance=True,
        scale_pos_weight=scale_pos_weight
    )

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(X_train), dtype=float)
    valid_proba_folds = []
    models = []
    vectorizers = []
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train_encoded), start=1):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train_encoded[tr_idx], y_train_encoded[va_idx]
        sent_tr = train_sentence.iloc[tr_idx]
        sent_va = train_sentence.iloc[va_idx]

        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
        X_tr_text = vectorizer.fit_transform(sent_tr)
        X_va_text = vectorizer.transform(sent_va)
        X_valid_text = vectorizer.transform(valid_sentence)

        X_tr_full = np.hstack([X_tr.to_numpy(), X_tr_text.toarray()])
        X_va_full = np.hstack([X_va.to_numpy(), X_va_text.toarray()])
        X_valid_full = np.hstack([X_valid.to_numpy(), X_valid_text.toarray()])

        feature_names = list(X_tr.columns) + [f"tfidf::{t}" for t in vectorizer.get_feature_names_out()]

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        callbacks = [lgb.early_stopping(stopping_rounds=200, verbose=False)]
        model.fit(
            X_tr_full,
            y_tr,
            eval_set=[(X_va_full, y_va)],
            eval_metric="auc",
            callbacks=callbacks,
        )

        va_proba = model.predict_proba(X_va_full)[:, 1]
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

        valid_proba_folds.append(model.predict_proba(X_valid_full)[:, 1])
        models.append((model, feature_names))
        vectorizers.append(vectorizer)

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

    # ---- Metrics ----
    acc = accuracy_score(y_valid_encoded, y_valid_pred_encoded)

    # Binary metrics for the positive class (=1)
    prec_pos = precision_score(y_valid_encoded, y_valid_pred_encoded, average='binary', pos_label=1, zero_division=0)
    rec_pos = recall_score(y_valid_encoded, y_valid_pred_encoded, average='binary', pos_label=1, zero_division=0)
    f1_pos = f1_score(y_valid_encoded, y_valid_pred_encoded, average='binary', pos_label=1, zero_division=0)

    # Macro-averaged metrics (average over class 0 and class 1)
    prec_macro = precision_score(y_valid_encoded, y_valid_pred_encoded, average='macro', zero_division=0)
    rec_macro = recall_score(y_valid_encoded, y_valid_pred_encoded, average='macro', zero_division=0)
    f1_macro = f1_score(y_valid_encoded, y_valid_pred_encoded, average='macro', zero_division=0)

    bal_acc = balanced_accuracy_score(y_valid_encoded, y_valid_pred_encoded)

    print("Accuracy:", acc)
    print("Precision (pos=1):", prec_pos)
    print("Recall (pos=1):", rec_pos)
    print("F1 (pos=1):", f1_pos)
    print("Precision (macro):", prec_macro)
    print("Recall (macro):", rec_macro)
    print("Macro F1:", f1_macro)
    print("Balanced Accuracy:", bal_acc)

    # ==========================
    # SHAP (right=SUCCESS(0), left=FAILURE(1))
    # Note: you said you flipped labels so that failure=1 and success=0.
    # We therefore visualize SHAP in the SUCCESS direction.
    # ==========================
    final_model, feature_names = models[-1]
    final_vectorizer = vectorizers[-1]
    X_valid_text_final = final_vectorizer.transform(valid_sentence)
    X_valid_full_final = np.hstack([X_valid.to_numpy(), X_valid_text_final.toarray()])

    explainer = shap.TreeExplainer(final_model.booster_, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X_valid_full_final, check_additivity=False)

    # shap_values can be either:
    # - a list of arrays (class-wise SHAP), e.g. [class0, class1]
    # - a single 2D array for the model output (usually the positive class (=1) margin)
    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=0)
    else:
        shap_array = shap_values

    print(f"SHAP values array shape: {getattr(shap_array, 'shape', None)}")

    # Ensure shape = (num_outputs, n_samples, n_features)
    if shap_array.ndim == 2:
        shap_array = shap_array[np.newaxis, ...]

    num_outputs = shap_array.shape[0]

    # We want: right = SUCCESS(0), left = FAILURE(1)
    # - If we have class-wise outputs (2 outputs), use class 0 directly.
    # - If we only have one output (typically for class 1 in binary), flip the sign to convert
    #   "push towards failure(1)" into "push towards success(0)".
    if num_outputs >= 2:
        shap_success = shap_array[0]
        print("[SHAP] Plotting class 0 (SUCCESS) contributions: right increases P(success=0).")
    else:
        shap_success = -shap_array[0]
        print("[SHAP] Single-output SHAP assumed for class 1 (FAILURE); sign-flipped to show SUCCESS direction.")

    shap.summary_plot(
        shap_success,
        X_valid_full_final,
        feature_names=feature_names,
        show=False,
        max_display=20
    )
    plt.title("SHAP summary (SUCCESS=0 to the right, FAILURE=1 to the left)")
    plt.tight_layout()
    
    # Optionally produce SHAP dependence plots for selected features.
    if create_shap_dependence:
        os.makedirs(shap_plot_dir, exist_ok=True)

        # shap_success: (n_samples, n_features)
        mean_abs_shap = np.mean(np.abs(shap_success), axis=0)
        top_idx = np.argsort(-mean_abs_shap)

        # Build list of indices to plot
        if dependence_features is None:
            selected_idx = top_idx[:max_dependence]
        else:
            selected_idx = []
            for f in dependence_features:
                if isinstance(f, int):
                    if 0 <= f < len(feature_names):
                        selected_idx.append(f)
                    else:
                        print(f"dependence feature index out of range: {f}")
                elif isinstance(f, str):
                    if f in feature_names:
                        selected_idx.append(feature_names.index(f))
                    else:
                        # try with tfidf:: prefix if absent
                        alt = f if f.startswith("tfidf::") else f"tfidf::{f}"
                        if alt in feature_names:
                            selected_idx.append(feature_names.index(alt))
                        else:
                            print(f"dependence feature name not found: {f}")
                else:
                    print(f"unsupported dependence feature type: {f}")

        # Remove duplicates while preserving order
        seen = set()
        selected_idx = [x for x in selected_idx if not (x in seen or seen.add(x))]

        # Log-transform selected numeric features for dependence plots only.
        log_targets = {"project_cost_plan", "project_duration_plan", "population"}
        X_valid_for_dependence = X_valid_full_final.copy()
        feature_names_for_plot = list(feature_names)
        for fname in log_targets:
            if fname in feature_names:
                fidx = feature_names.index(fname)
                col = np.asarray(X_valid_for_dependence[:, fidx], dtype=float)
                col = np.where(col >= 0, np.log1p(col), np.nan)
                X_valid_for_dependence[:, fidx] = col
                feature_names_for_plot[fidx] = f"log1p::{fname}"

        for idx in selected_idx:
            fname = feature_names_for_plot[idx]
            try:
                shap.dependence_plot(
                    idx,
                    shap_success,
                    X_valid_for_dependence,
                    feature_names=feature_names_for_plot,
                    show=False,
                )
                fig = plt.gcf()
                safe_name = fname.replace('/', '_').replace(' ', '_')
                out_path = os.path.join(shap_plot_dir, f"shap_dependence_{idx}_{safe_name}.png")
                if save_shap_plots:
                    fig.savefig(out_path, bbox_inches='tight')
                    print(f"Saved SHAP dependence plot: {out_path}")
                plt.close(fig)
            except Exception as e:
                print(f"Failed to create dependence plot for {fname} (idx={idx}): {e}")

    plt.show()
