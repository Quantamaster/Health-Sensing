import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os

def load_data(dataset_dir):
    data = []
    labels = []
    for file in os.listdir(dataset_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(dataset_dir, file))
            data.append(np.array(df[['nasal_airflow', 'thoracic_movement', 'spo2']].tolist()))
            labels.append(df['label'].values)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    return data, labels

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_conv_lstm_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, cm

def main(dataset_dir):
    data, labels = load_data(dataset_dir)
    label_encoder = {label: idx for idx, label in enumerate(np.unique(labels))}
    labels = np.array([label_encoder[label] for label in labels])

    loo = LeaveOneOut()
    for train_index, test_index in loo.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = len(np.unique(y_train))

        cnn_model = build_cnn_model(input_shape, num_classes)
        cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        conv_lstm_model = build_conv_lstm_model(input_shape, num_classes)
        conv_lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        cnn_accuracy, cnn_precision, cnn_recall, cnn_cm = evaluate_model(cnn_model, X_test, y_test)
        conv_lstm_accuracy, conv_lstm_precision, conv_lstm_recall, conv_lstm_cm = evaluate_model(conv_lstm_model, X_test, y_test)

        print(f'CNN Model - Accuracy: {cnn_accuracy}, Precision: {cnn_precision}, Recall: {cnn_recall}')
        print(f'Conv-LSTM Model - Accuracy: {conv_lstm_accuracy}, Precision: {conv_lstm_precision}, Recall: {conv_lstm_recall}')

if __name__ == '__main__':
    dataset_dir = 'Dataset'
    main(dataset_dir)
