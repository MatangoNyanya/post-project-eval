import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import shap
import matplotlib.pyplot as plt

def train_and_evaluate_model(train_df, valid_df):

     # データのロードと前処理
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_valid = valid_df.drop(columns=['label'])
    y_valid = valid_df['label']

    all_labels = train_df['label'].tolist() + valid_df['label'].tolist()
    unique_labels = np.unique(all_labels)
    
    # ラベルエンコーディングの設定
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valid_encoded = label_encoder.transform(y_valid)
    
    # モデルのトレーニングとSHAP解析
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train_encoded)
    
    # 予測と評価
    predictions_df = pd.DataFrame({
        'label': y_valid,
        'predicted_label': label_encoder.inverse_transform(model.predict(X_valid)),
        **valid_df.drop(columns=['label']).to_dict('series')
    })
    predictions_df.to_csv('results/classification/result_num_xgb.csv', index=False)
    conf_matrix_df = pd.DataFrame(confusion_matrix(predictions_df['label'], predictions_df['predicted_label']),
                                  columns=sorted(set(predictions_df['label']) | set(predictions_df['predicted_label'])),
                                  index=sorted(set(predictions_df['label']) | set(predictions_df['predicted_label'])))
    conf_matrix_df.to_csv('results/classification/confusion_matrix_num_xgb.csv')
    
    # メトリクスの計算と表示
    print("Accuracy:", accuracy_score(predictions_df['label'], predictions_df['predicted_label']))
    print("Precision:", precision_score(predictions_df['label'], predictions_df['predicted_label'], average='macro'))
    print("Recall:", recall_score(predictions_df['label'], predictions_df['predicted_label'], average='macro'))
    
    # SHAP解析とプロット
    explainer = shap.Explainer(model)
    shap_values = explainer(X_valid)
    
    print(len(unique_labels))
    
    feature_names=X_train.columns
    
    
    # # 2クラスのとき
    if len(unique_labels)==2:
        shap.summary_plot(shap_values, X_valid,  feature_names=feature_names)
    else:
        # 3クラス以上のとき
        for i in range(len(unique_labels)):
            print("Class:" + str(i))  # 修正: iを文字列に変換して出力

            shap.summary_plot(shap_values[:, :, i], X_valid,  feature_names=feature_names)
    
    
    
    plt.show()

