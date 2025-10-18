import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_fscore_support

def vis(confusion_matrix_path, results_path):
    # Confusion matrix の読み込み
    confusion_matrix_df = pd.read_csv(confusion_matrix_path, index_col=0)
    
    # ヒートマップの描画
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=True, square=True)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    # 予測結果の読み込み
    results_df = pd.read_csv(results_path)

    # 精度評価の計算
    true_labels = results_df['label']
    predicted_labels = results_df['predicted_label']

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')

    print("method, class, accuracy, precision, recall")
    print(f"proposed, all, {accuracy}, {precision}, {recall}")

    # クラスごとのPrecisionとRecall
    unique_labels = sorted(true_labels.unique())
    precisions, recalls, _, _ = precision_recall_fscore_support(true_labels, predicted_labels, labels=unique_labels)
    for label, precision, recall in zip(unique_labels, precisions, recalls):
        print(f"proposed, {label}, non, {precision}, {recall}")

    # ランダム予測のベースラインの計算
    label_counts = true_labels.value_counts(normalize=True).sort_index()
    baseline_precisions = label_counts.values
    baseline_recalls = label_counts.values

    baseline_accuracy = np.mean([label_counts[label] for label in true_labels])

    print(f"baseline, all, {baseline_accuracy}, {np.mean(baseline_precisions)}, {np.mean(baseline_recalls)}")

    # ベースラインのクラスごとのPrecisionとRecall
    for label, baseline_precision, baseline_recall in zip(unique_labels, baseline_precisions, baseline_recalls):
        print(f"baseline, {label}, non, {baseline_precision}, {baseline_recall}")
