import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, vstack
from typing import Tuple

def load_numerical_data(train_path: str, valid_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data = pd.read_csv(train_path)
    validation_data = pd.read_csv(valid_path)
    return train_data, validation_data

def preprocess_numerical_data(train_data: pd.DataFrame, validation_data: pd.DataFrame) -> Tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray, LabelEncoder]:
    all_labels = pd.concat([train_data['target'], validation_data['target']])

    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    X_val = validation_data.drop(columns=['target'])
    y_val = validation_data['target']

    label_encoder = LabelEncoder()
    all_labels_encoded = label_encoder.fit_transform(all_labels)
    encoded_labels_train = label_encoder.transform(y_train)
    encoded_labels_valid = label_encoder.transform(y_val)

    missing_labels = set(all_labels_encoded) - set(encoded_labels_train)
    for label in missing_labels:
        encoded_labels_train = np.append(encoded_labels_train, label)
        missing_label_features = X_train.mean(axis=0).values.reshape(1, -1)
        missing_label_features_sparse = csr_matrix(missing_label_features)
        X_train = vstack([csr_matrix(X_train), missing_label_features_sparse])

    return csr_matrix(X_train), encoded_labels_train, csr_matrix(X_val), encoded_labels_valid, label_encoder

def train_and_evaluate_numerical_model(model, X_train: csr_matrix, X_val: csr_matrix, encoded_labels_train: np.ndarray, y_val: np.ndarray, validation_data: pd.DataFrame, label_encoder: LabelEncoder, output_prefix: str):
    model.fit(X_train, encoded_labels_train)
    valid_predictions = model.predict_proba(X_val)
    predicted_labels = np.argmax(valid_predictions, axis=1)
    original_valid_predictions = label_encoder.inverse_transform(predicted_labels)

    predictions_df = pd.DataFrame({
        'label': y_val,
        'predicted_label': original_valid_predictions,
        **validation_data.drop(columns=['target']).to_dict('series')
    })
    predictions_df.to_csv(f'results/classification/result_{output_prefix}.csv', index=False)

    conf_matrix = confusion_matrix(predictions_df['label'], predictions_df['predicted_label'])
    unique_labels = sorted(set(predictions_df['label'].unique()) | set(predictions_df['predicted_label'].unique()))
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=unique_labels, index=unique_labels)
    conf_matrix_df.to_csv(f'results/classification/confusion_matrix_{output_prefix}.csv')

    accuracy = accuracy_score(predictions_df['label'], predictions_df['predicted_label'])
    precision = precision_score(predictions_df['label'], predictions_df['predicted_label'], average='weighted')
    recall = recall_score(predictions_df['label'], predictions_df['predicted_label'], average='weighted')

    print(f"■{output_prefix}")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

def predict():
    train_data, validation_data = load_numerical_data('dataset/train_num.csv', 'dataset/validation_num.csv')
    X_train, encoded_labels_train, X_val, encoded_labels_valid, label_encoder = preprocess_numerical_data(train_data, validation_data)

    # Train and evaluate RandomForest
    rf_model = RandomForestClassifier(random_state=42)
    train_and_evaluate_numerical_model(rf_model, X_train, X_val, encoded_labels_train, encoded_labels_valid, validation_data, label_encoder, "num")
