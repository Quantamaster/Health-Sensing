"""
DeepMedico Machine Learning Models
1D CNN and Conv-LSTM for sleep breathing irregularity detection
Usage: python modeling.py --dataset "Dataset/sleep_breathing_dataset.parquet"
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class SleepBreathingClassifier:
    def __init__(self, model_type='1d_cnn', n_classes=3, input_shape=(2040,)):
        self.model_type = model_type
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def build_1d_cnn_model(self):
        """Build 1D CNN architecture"""
        model = models.Sequential([
            # Reshape for 1D CNN (treat features as time series)
            layers.Reshape((2040, 1), input_shape=self.input_shape),

            # First convolutional block
            layers.Conv1D(64, kernel_size=7, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),

            # Second convolutional block
            layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),

            # Third convolutional block
            layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.5),

            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.n_classes, activation='softmax')
        ])

        return model

    def build_conv_lstm_model(self):
        """Build Conv-LSTM hybrid architecture"""
        # Reshape data for Conv-LSTM
        # We'll treat the data as sequences of multi-channel signals

        model = models.Sequential([
            # Reshape to (timesteps, features) - treating every 60 samples as timestep
            layers.Reshape((34, 60), input_shape=self.input_shape),

            # Convolutional layers for feature extraction
            layers.Conv1D(64, kernel_size=7, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),

            # More conv layers
            layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),

            # LSTM layers for temporal modeling
            layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),

            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.n_classes, activation='softmax')
        ])

        return model

    def build_model(self):
        """Build model based on specified type"""
        if self.model_type == '1d_cnn':
            self.model = self.build_1d_cnn_model()
        elif self.model_type == 'conv_lstm':
            self.model = self.build_conv_lstm_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train the model"""

        # Prepare callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                patience=15,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ]

        # Fit model
        validation_data = (X_val, y_val) if X_val is not None else None

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks_list,
            verbose=1
        )

        return history

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Evaluate model performance"""
        y_pred_proba = self.predict(X)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average=None)

        # Calculate specificity for each class
        cm = confusion_matrix(y, y_pred)
        specificity = []
        for i in range(len(cm)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': recall,  # Recall is the same as sensitivity
            'specificity': np.array(specificity),
            'f1_score': f1,
            'support': support,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }


def load_dataset(dataset_path):
    """Load dataset from file"""
    if dataset_path.endswith('.parquet'):
        df = pd.read_parquet(dataset_path)
        print(f"Loaded dataset from Parquet: {df.shape}")
    elif dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset from CSV: {df.shape}")
    else:
        raise ValueError("Unsupported dataset format. Use .parquet or .csv")

    # Extract features and labels
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['label'].values
    participants = df['participant_id'].values

    print(f"Features: {X.shape}")
    print(f"Labels distribution: {pd.Series(y).value_counts().to_dict()}")
    print(f"Participants: {len(np.unique(participants))}")

    return X, y, participants


def calculate_class_weights(y):
    """Calculate class weights for imbalanced dataset"""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    print(f"Class weights: {class_weight_dict}")
    return class_weight_dict


def cross_validate_model(X, y, participants, model_type='1d_cnn', n_epochs=50):
    """Perform leave-one-subject-out cross-validation"""

    # Initialize results storage
    fold_results = []
    all_predictions = []
    all_true_labels = []
    all_fold_info = []

    # Set up Leave-One-Group-Out cross-validation
    logo = LeaveOneGroupOut()

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)

    print(f"Classes: {label_encoder.classes_}")
    print(f"Starting {logo.get_n_splits(X, y_encoded, participants)}-fold cross-validation")

    fold_idx = 0
    for train_idx, test_idx in logo.split(X, y_encoded, participants):
        fold_idx += 1
        test_participant = participants[test_idx[0]]
        print(f"\n=== Fold {fold_idx}: Testing on {test_participant} ===")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Calculate class weights
        class_weights = calculate_class_weights(y_train)

        # Build and train model
        classifier = SleepBreathingClassifier(
            model_type=model_type,
            n_classes=n_classes,
            input_shape=(X_train_scaled.shape[1],)
        )

        classifier.build_model()
        print(f"Model architecture: {model_type}")

        # Train model
        history = classifier.fit(
            X_train_scaled, y_train,
            epochs=n_epochs,
            batch_size=32
        )

        # Evaluate model
        results = classifier.evaluate(X_test_scaled, y_test)
        results['test_participant'] = test_participant
        results['fold'] = fold_idx

        # Store results
        fold_results.append(results)
        all_predictions.extend(results['predictions'])
        all_true_labels.extend(y_test)
        all_fold_info.extend([test_participant] * len(y_test))

        # Print fold results
        print(f"Fold {fold_idx} Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"  {class_name}:")
            print(f"    Precision: {results['precision'][i]:.4f}")
            print(f"    Recall: {results['recall'][i]:.4f}")
            print(f"    Specificity: {results['specificity'][i]:.4f}")

        print(f"  Confusion Matrix:")
        print(results['confusion_matrix'])

        # Clean up memory
        del classifier
        tf.keras.backend.clear_session()

    return fold_results, label_encoder, all_predictions, all_true_labels, all_fold_info


def aggregate_results(fold_results, label_encoder):
    """Aggregate cross-validation results"""
    n_classes = len(label_encoder.classes_)

    # Collect metrics across folds
    accuracies = [result['accuracy'] for result in fold_results]
    precisions = np.array([result['precision'] for result in fold_results])
    recalls = np.array([result['recall'] for result in fold_results])
    specificities = np.array([result['specificity'] for result in fold_results])
    f1_scores = np.array([result['f1_score'] for result in fold_results])

    # Calculate mean and std
    aggregated_results = {
        'accuracy': {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'per_fold': accuracies
        }
    }

    # Per-class metrics
    for i, class_name in enumerate(label_encoder.classes_):
        aggregated_results[class_name] = {
            'precision': {
                'mean': np.mean(precisions[:, i]),
                'std': np.std(precisions[:, i]),
                'per_fold': precisions[:, i].tolist()
            },
            'recall': {
                'mean': np.mean(recalls[:, i]),
                'std': np.std(recalls[:, i]),
                'per_fold': recalls[:, i].tolist()
            },
            'sensitivity': {
                'mean': np.mean(recalls[:, i]),
                'std': np.std(recalls[:, i]),
                'per_fold': recalls[:, i].tolist()
            },
            'specificity': {
                'mean': np.mean(specificities[:, i]),
                'std': np.std(specificities[:, i]),
                'per_fold': specificities[:, i].tolist()
            },
            'f1_score': {
                'mean': np.mean(f1_scores[:, i]),
                'std': np.std(f1_scores[:, i]),
                'per_fold': f1_scores[:, i].tolist()
            }
        }

    return aggregated_results


def save_results(results, model_type, output_dir='Results'):
    """Save results to files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save aggregated results as JSON
    results_file = os.path.join(output_dir, f'{model_type}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {results_file}")


def print_summary_results(aggregated_results, model_type):
    """Print formatted summary of results"""
    print(f"\n{'=' * 60}")
    print(f"SUMMARY RESULTS - {model_type.upper()}")
    print(f"{'=' * 60}")

    print(
        f"\nOverall Accuracy: {aggregated_results['accuracy']['mean']:.4f} ± {aggregated_results['accuracy']['std']:.4f}")

    print(f"\nPer-Class Performance:")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'Specificity':<12} {'F1-Score':<12}")
    print(f"{'-' * 80}")

    for class_name in ['Normal', 'Hypopnea', 'Obstructive Apnea']:
        if class_name in aggregated_results:
            prec = aggregated_results[class_name]['precision']['mean']
            prec_std = aggregated_results[class_name]['precision']['std']
            rec = aggregated_results[class_name]['recall']['mean']
            rec_std = aggregated_results[class_name]['recall']['std']
            spec = aggregated_results[class_name]['specificity']['mean']
            spec_std = aggregated_results[class_name]['specificity']['std']
            f1 = aggregated_results[class_name]['f1_score']['mean']
            f1_std = aggregated_results[class_name]['f1_score']['std']

            print(
                f"{class_name:<20} {prec:.3f}±{prec_std:.3f}   {rec:.3f}±{rec_std:.3f}   {spec:.3f}±{spec_std:.3f}   {f1:.3f}±{f1_std:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate sleep breathing models')
    parser.add_argument('--dataset', required=True, help='Path to dataset file')
    parser.add_argument('--model', choices=['1d_cnn', 'conv_lstm', 'both'],
                        default='both', help='Model type to train')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--output_dir', default='Results',
                        help='Output directory for results')

    args = parser.parse_args()

    print("DeepMedico™ Sleep Breathing Irregularity Detection")
    print("=" * 55)

    # Load dataset
    X, y, participants = load_dataset(args.dataset)

    # Determine models to train
    if args.model == 'both':
        models_to_train = ['1d_cnn', 'conv_lstm']
    else:
        models_to_train = [args.model]

    # Train and evaluate models
    for model_type in models_to_train:
        print(f"\nTraining {model_type.upper()} model...")

        # Perform cross-validation
        fold_results, label_encoder, all_predictions, all_true_labels, all_fold_info = cross_validate_model(
            X, y, participants, model_type=model_type, n_epochs=args.epochs
        )

        # Aggregate results
        aggregated_results = aggregate_results(fold_results, label_encoder)

        # Print summary
        print_summary_results(aggregated_results, model_type)

        # Save results
        save_results(aggregated_results, model_type, args.output_dir)

    print(f"\nAll results saved to: {args.output_dir}")
    print("Training complete!")


if __name__ == "__main__":
    main()
