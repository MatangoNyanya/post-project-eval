import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
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

def train_and_evaluate_model(train_df, valid_df, model_name):

    # データのロードと前処理
    train_df = train_df.copy()
    valid_df = valid_df.copy()
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
        
    # モデルのトレーニングとSHAP解析
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_train, y_train_encoded)
    
    # 予測と評価
    y_valid_pred_encoded = model.predict(X_valid)

    predictions_df = pd.DataFrame({
        'label': y_valid,
        'predicted_label': label_encoder.inverse_transform(y_valid_pred_encoded),
        'sentence': valid_df['sentence']
    })
    predictions_df.to_csv('results/classification/result_text_lgb.csv', index=False)
    conf_matrix_df = pd.DataFrame(confusion_matrix(predictions_df['label'], predictions_df['predicted_label']),
                                  columns=sorted(set(predictions_df['label']) | set(predictions_df['predicted_label'])),
                                  index=sorted(set(predictions_df['label']) | set(predictions_df['predicted_label'])))
    conf_matrix_df.to_csv('results/classification/confusion_matrix_text_lgb.csv')
    
    # メトリクスの計算と表示
    print("Accuracy:", accuracy_score(y_valid_encoded, y_valid_pred_encoded))
    print("Precision:", precision_score(y_valid_encoded, y_valid_pred_encoded, average='macro', zero_division=0))
    print("Recall:", recall_score(y_valid_encoded, y_valid_pred_encoded, average='macro', zero_division=0))
    print("Macro F1:", f1_score(y_valid_encoded, y_valid_pred_encoded, average='macro', zero_division=0))
    print("Balanced Accuracy:", balanced_accuracy_score(y_valid_encoded, y_valid_pred_encoded))


    # SHAP解析とプロット
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_valid)
    
    print(len(unique_labels))
    
    # 2クラスのとき
    if len(unique_labels) == 2:
        shap.summary_plot(shap_values, X_valid, feature_names=vectorizer.get_feature_names_out())
    else:
        # 3クラス以上のとき
        for i in range(len(unique_labels)):
            print("Class:" + str(i))  # 修正: iを文字列に変換して出力

            shap.summary_plot(shap_values[:, :, i], X_valid, feature_names=vectorizer.get_feature_names_out())
    
    plt.show()
