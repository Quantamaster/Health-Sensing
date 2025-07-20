"""
DeepMedico Bonus: Sleep Stage Classification
Uses same framework but replaces breathing event labels with sleep stage labels
Usage: python sleep_stage_classification.py -in_dir "Data" -out_dir "SleepStageDataset" --train
"""

import argparse
import os
import pandas as pd
import numpy as np
from scipy import signal
import json
import warnings

warnings.filterwarnings('ignore')


def bandpass_filter(data, fs, lowcut=0.17, highcut=0.4, order=3):
    """Apply bandpass filter to remove noise outside breathing frequency range"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    sos = signal.butter(order, [low, high], btype='band', output='sos')
    filtered_data = signal.sosfilt(sos, data)

    return filtered_data


def load_and_preprocess_signals(participant_folder):
    """Load and preprocess signal data with filtering"""
    signals = {}

    signal_files = {
        'nasal_airflow': {'rate': 32, 'file': 'nasal_airflow.csv'},
        'thoracic_movement': {'rate': 32, 'file': 'thoracic_movement.csv'},
        'spo2': {'rate': 4, 'file': 'spo2.csv'}
    }

    for signal_name, info in signal_files.items():
        file_path = os.path.join(participant_folder, info['file'])
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df.columns = ['timestamp', 'value']
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

                if signal_name in ['nasal_airflow', 'thoracic_movement']:
                    df['value'] = bandpass_filter(df['value'].values, info['rate'])
                    print(f"Applied bandpass filter to {signal_name}")

                signals[signal_name] = df
                print(f"Loaded and processed {signal_name}: {len(df)} samples")

            except Exception as e:
                print(f"Error loading {signal_name}: {e}")
        else:
            print(f"File not found: {file_path}")

    return signals


def load_sleep_stages(participant_folder):
    """Load sleep stage annotations"""
    sleep_profile_file = os.path.join(participant_folder, 'sleep_profile.csv')
    sleep_stages = []

    if os.path.exists(sleep_profile_file):
        try:
            df = pd.read_csv(sleep_profile_file)
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])

            # Filter for relevant sleep stages
            relevant_stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
            df = df[df['sleep_stage'].isin(relevant_stages)]

            sleep_stages = df.to_dict('records')
            print(f"Loaded {len(sleep_stages)} sleep stage annotations")

        except Exception as e:
            print(f"Error loading sleep stages: {e}")
    else:
        print(f"Sleep profile file not found: {sleep_profile_file}")

    return sleep_stages


def create_windows(signals, window_size_sec=30, overlap_ratio=0.5):
    """Create overlapping windows from continuous signals"""

    start_times = [df.index.min() for df in signals.values()]
    end_times = [df.index.max() for df in signals.values()]

    common_start = max(start_times)
    common_end = min(end_times)

    step_size_sec = window_size_sec * (1 - overlap_ratio)

    current_time = common_start
    windows = []

    while current_time + pd.Timedelta(seconds=window_size_sec) <= common_end:
        window_start = current_time
        window_end = current_time + pd.Timedelta(seconds=window_size_sec)

        windows.append({
            'start_time': window_start,
            'end_time': window_end,
            'window_id': len(windows)
        })

        current_time += pd.Timedelta(seconds=step_size_sec)

    print(f"Created {len(windows)} windows from {common_start} to {common_end}")
    return windows


def assign_sleep_stage_labels(windows, sleep_stages):
    """Assign sleep stage labels to windows based on temporal overlap"""
    labeled_windows = []

    for window in windows:
        window_duration = (window['end_time'] - window['start_time']).total_seconds()
        best_label = 'Wake'  # Default to Wake if no overlap found
        max_overlap_ratio = 0

        for stage in sleep_stages:
            # Calculate overlap
            overlap_start = max(window['start_time'], stage['start_time'])
            overlap_end = min(window['end_time'], stage['end_time'])

            if overlap_start < overlap_end:
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                overlap_ratio = overlap_duration / window_duration

                # Assign stage label if overlap is significant
                if overlap_ratio > max_overlap_ratio:
                    best_label = stage['sleep_stage']
                    max_overlap_ratio = overlap_ratio

        window['label'] = best_label
        labeled_windows.append(window)

    # Print label distribution
    label_counts = {}
    for window in labeled_windows:
        label = window['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"Sleep stage distribution: {label_counts}")
    return labeled_windows


def extract_window_data(signals, window):
    """Extract signal data for a specific window"""
    window_data = {}

    for signal_name, df in signals.items():
        mask = (df.index >= window['start_time']) & (df.index < window['end_time'])
        window_signal = df.loc[mask, 'value'].values

        if signal_name == 'spo2':
            target_samples = int(30 * 4)
        else:
            target_samples = int(30 * 32)

        if len(window_signal) != target_samples and len(window_signal) > 0:
            from scipy import interpolate
            x_old = np.linspace(0, 1, len(window_signal))
            x_new = np.linspace(0, 1, target_samples)
            f = interpolate.interp1d(x_old, window_signal, kind='linear',
                                     fill_value='extrapolate')
            window_signal = f(x_new)
        elif len(window_signal) == 0:
            window_signal = np.zeros(target_samples)

        window_data[signal_name] = window_signal

    return window_data


def create_feature_matrix(signals, labeled_windows):
    """Create feature matrix from windowed signals"""

    features = []
    labels = []
    window_info = []

    for window in labeled_windows:
        try:
            window_data = extract_window_data(signals, window)

            feature_vector = []

            if 'nasal_airflow' in window_data:
                feature_vector.extend(window_data['nasal_airflow'])
            else:
                feature_vector.extend(np.zeros(960))

            if 'thoracic_movement' in window_data:
                feature_vector.extend(window_data['thoracic_movement'])
            else:
                feature_vector.extend(np.zeros(960))

            if 'spo2' in window_data:
                feature_vector.extend(window_data['spo2'])
            else:
                feature_vector.extend(np.zeros(120))

            features.append(feature_vector)
            labels.append(window['label'])

            window_info.append({
                'window_id': window['window_id'],
                'start_time': window['start_time'],
                'end_time': window['end_time'],
                'label': window['label']
            })

        except Exception as e:
            print(f"Error processing window {window['window_id']}: {e}")
            continue

    print(f"Created feature matrix: {len(features)} samples x {len(features[0]) if features else 0} features")
    return np.array(features), np.array(labels), window_info


def process_participant_sleep_stages(participant_folder, participant_id):
    """Process a single participant's data for sleep stage classification"""
    print(f"\nProcessing participant for sleep stages: {participant_id}")

    signals = load_and_preprocess_signals(participant_folder)
    if not signals:
        print(f"No signals found for participant {participant_id}")
        return None

    sleep_stages = load_sleep_stages(participant_folder)
    if not sleep_stages:
        print(f"No sleep stages found for participant {participant_id}")
        return None

    windows = create_windows(signals)
    labeled_windows = assign_sleep_stage_labels(windows, sleep_stages)
    features, labels, window_info = create_feature_matrix(signals, labeled_windows)

    if len(features) == 0:
        print(f"No valid features created for participant {participant_id}")
        return None

    return {
        'participant_id': participant_id,
        'features': features,
        'labels': labels,
        'window_info': window_info,
        'signal_info': {
            'nasal_airflow_samples': 960,
            'thoracic_movement_samples': 960,
            'spo2_samples': 120,
            'total_features': 2040
        }
    }


def save_sleep_stage_dataset(dataset, output_dir):
    """Save sleep stage dataset"""
    os.makedirs(output_dir, exist_ok=True)

    all_features = []
    all_labels = []
    all_participants = []
    all_window_info = []

    for participant_data in dataset:
        n_samples = len(participant_data['features'])
        all_features.extend(participant_data['features'].tolist())
        all_labels.extend(participant_data['labels'].tolist())
        all_participants.extend([participant_data['participant_id']] * n_samples)
        all_window_info.extend(participant_data['window_info'])

    # Create DataFrame
    df = pd.DataFrame(all_features)
    df.columns = [f'feature_{i}' for i in range(df.shape[1])]
    df['label'] = all_labels
    df['participant_id'] = all_participants

    # Add window info
    for i, info in enumerate(all_window_info):
        df.loc[i, 'window_id'] = info['window_id']
        df.loc[i, 'start_time'] = info['start_time']
        df.loc[i, 'end_time'] = info['end_time']

    # Save as Parquet
    parquet_path = os.path.join(output_dir, 'sleep_stage_dataset.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"Sleep stage dataset saved: {parquet_path}")

    # Save metadata
    metadata = {
        'total_samples': len(df),
        'participants': [p['participant_id'] for p in dataset],
        'feature_info': dataset[0]['signal_info'] if dataset else {},
        'label_distribution': df['label'].value_counts().to_dict(),
        'creation_timestamp': pd.Timestamp.now().isoformat(),
        'task_type': 'sleep_stage_classification'
    }

    metadata_path = os.path.join(output_dir, 'sleep_stage_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Create sleep stage classification dataset')
    parser.add_argument('-in_dir', '--input_dir', required=True,
                        help='Input directory containing participant folders')
    parser.add_argument('-out_dir', '--output_dir', required=True,
                        help='Output directory for sleep stage dataset')
    parser.add_argument('--train', action='store_true',
                        help='Also train models on the created dataset')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    print("DeepMedico™ Bonus: Sleep Stage Classification")
    print("=" * 50)
    print(f"Creating sleep stage dataset from: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Find participant folders
    participant_folders = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            participant_folders.append((item_path, item))

    print(f"Found {len(participant_folders)} participant folders")

    # Process each participant
    dataset = []
    for participant_folder, participant_id in participant_folders:
        participant_data = process_participant_sleep_stages(participant_folder, participant_id)
        if participant_data:
            dataset.append(participant_data)

    if not dataset:
        print("No valid participant data found!")
        return

    print(f"\nSleep stage dataset creation complete!")
    print(f"Total participants: {len(dataset)}")
    total_samples = sum(len(p['features']) for p in dataset)
    print(f"Total samples: {total_samples}")

    # Save dataset
    save_sleep_stage_dataset(dataset, output_dir)

    # Train models if requested
    if args.train:
        print("\nTraining models on sleep stage dataset...")
        dataset_path = os.path.join(output_dir, 'sleep_stage_dataset.parquet')

        # Import and run the modeling script
        from modeling import load_dataset, cross_validate_model, aggregate_results, save_results, print_summary_results

        # Load the sleep stage dataset
        X, y, participants = load_dataset(dataset_path)

        # Train both models
        models_to_train = ['1d_cnn', 'conv_lstm']

        for model_type in models_to_train:
            print(f"\nTraining {model_type.upper()} model for sleep stage classification...")

            # Perform cross-validation
            fold_results, label_encoder, all_predictions, all_true_labels, all_fold_info = cross_validate_model(
                X, y, participants, model_type=model_type, n_epochs=50
            )

            # Aggregate results
            aggregated_results = aggregate_results(fold_results, label_encoder)

            # Print summary
            print_summary_results(aggregated_results, f"{model_type}_sleep_stage")

            # Save results
            save_results(aggregated_results, f"{model_type}_sleep_stage", output_dir)

    print(f"\nAll sleep stage results saved to: {output_dir}")
    print("Sleep stage classification complete!")


if __name__ == "__main__":
    main()
