import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from sklearn.ensemble import RandomForestClassifier
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

def train_and_evaluate_model(train_df, valid_df, model_name):

    # データのロードと前処理
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    train_df['sentence'] = train_df['sentence'].fillna('').astype(str)
    valid_df['sentence'] = valid_df['sentence'].fillna('').astype(str)

    if train_df['label'].isna().any() or valid_df['label'].isna().any():
        raise ValueError("label 列に欠損値が含まれています。欠損を除去または補完してください。")

    all_sentences = train_df['sentence'].tolist() + valid_df['sentence'].tolist()
    all_labels = train_df['label'].tolist() + valid_df['label'].tolist()
    unique_labels = np.unique(all_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_sentences = [' '.join(tokenizer.tokenize(sentence)) for sentence in all_sentences]
    
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    X = vectorizer.fit_transform(tokenized_sentences)
    X_train, X_valid = X[:len(train_df)].toarray(), X[len(train_df):].toarray()
    y_train, y_valid = np.array(all_labels[:len(train_df)]), np.array(all_labels[len(train_df):])
    
    # ラベルエンコーディングの設定
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valid_encoded = label_encoder.transform(y_valid)
    
    # モデルのトレーニングとSHAP解析
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train_encoded)
    
    # 予測と評価
    y_valid_pred_encoded = model.predict(X_valid)

    predictions_df = pd.DataFrame({
        'label': y_valid,
        'predicted_label': label_encoder.inverse_transform(y_valid_pred_encoded),
        'sentence': valid_df['sentence']
    })
    predictions_df.to_csv('results/classification/result_text_rf.csv', index=False)
    conf_matrix_df = pd.DataFrame(confusion_matrix(predictions_df['label'], predictions_df['predicted_label']),
                                  columns=sorted(set(predictions_df['label']) | set(predictions_df['predicted_label'])),
                                  index=sorted(set(predictions_df['label']) | set(predictions_df['predicted_label'])))
    conf_matrix_df.to_csv('results/classification/confusion_matrix_text_rf.csv')
    
    # メトリクスの計算と表示
    print("Accuracy:", accuracy_score(y_valid_encoded, y_valid_pred_encoded))
    print("Precision:", precision_score(y_valid_encoded, y_valid_pred_encoded, average='macro', zero_division=0))
    print("Recall:", recall_score(y_valid_encoded, y_valid_pred_encoded, average='macro', zero_division=0))
    print("Macro F1:", f1_score(y_valid_encoded, y_valid_pred_encoded, average='macro', zero_division=0))
    print("Balanced Accuracy:", balanced_accuracy_score(y_valid_encoded, y_valid_pred_encoded))

    
    # SHAP解析とプロット
    explainer = shap.TreeExplainer(model, feature_perturbation='interventional')  # ここで 'interventional' を指定
    shap_values = explainer.shap_values(X_valid, check_additivity=False)  # check_additivity=False を追加

    # SHAP値を配列に揃える（shapがリストを返すケースに対応）
    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=0)  # (n_classes, n_samples, n_features)
    else:
        shap_array = shap_values  # (n_samples, n_features) または (n_classes, n_samples, n_features)

    print(f"SHAP values array shape: {shap_array.shape}")

    # 配列の次元数を正規化
    if shap_array.ndim == 2:
        shap_array = shap_array[np.newaxis, ...]  # (1, n_samples, n_features)

    # 3クラス以上の場合のSHAPプロット
    num_classes = shap_array.shape[0]
    feature_names = vectorizer.get_feature_names_out()
    if num_classes == 1:
        shap.summary_plot(shap_array[0], X_valid, feature_names=feature_names)
    elif num_classes == 2:
        shap.summary_plot(shap_array[1], X_valid, feature_names=feature_names)
    else:
        for i in range(num_classes):
            print(f"Class: {i}")
            shap.summary_plot(shap_array[i], X_valid, feature_names=feature_names)
        
    plt.show()
