import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)
import shap
import matplotlib.pyplot as plt


def train_and_evaluate_model(train_df, valid_df):

    # データのロードと前処理
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

    # ラベルエンコーディングの設定
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
        is_unbalance=True,  # クラス不均衡設定
    )

    # モデルのトレーニング
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_train, y_train_encoded)

    # 予測と評価
    y_valid_pred_encoded = model.predict(X_valid)

    predictions_df = pd.DataFrame({
        'label': y_valid,
        'predicted_label': label_encoder.inverse_transform(y_valid_pred_encoded),
        **valid_df.drop(columns=['label']).to_dict('series')
    })
    predictions_df.to_csv('results/classification/result_num_lgb.csv', index=False)
    conf_matrix_df = pd.DataFrame(
        confusion_matrix(predictions_df['label'], predictions_df['predicted_label']),
        columns=sorted(set(predictions_df['label']) | set(predictions_df['predicted_label'])),
        index=sorted(set(predictions_df['label']) | set(predictions_df['predicted_label']))
    )
    conf_matrix_df.to_csv('results/classification/confusion_matrix_num_lgb.csv')

    # メトリクスの計算と表示
    print("Accuracy:", accuracy_score(y_valid_encoded, y_valid_pred_encoded))
    print("Precision:", precision_score(y_valid_encoded, y_valid_pred_encoded, average='macro', zero_division=0))
    print("Recall:", recall_score(y_valid_encoded, y_valid_pred_encoded, average='macro', zero_division=0))
    print("Macro F1:", f1_score(y_valid_encoded, y_valid_pred_encoded, average='macro', zero_division=0))
    print("Balanced Accuracy:", balanced_accuracy_score(y_valid_encoded, y_valid_pred_encoded))

    # SHAP解析とプロット
    explainer = shap.TreeExplainer(model.booster_, feature_perturbation='interventional')
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
