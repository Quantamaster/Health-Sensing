"""
DeepMedico Dataset Creation Script
Creates labeled 30-second windows from preprocessed sleep data
Usage: python create_dataset.py -in_dir "Data" -out_dir "Dataset"
"""

import argparse
import os
import pandas as pd
import numpy as np
from scipy import signal
import pickle
import warnings

warnings.filterwarnings('ignore')


def bandpass_filter(data, fs, lowcut=0.17, highcut=0.4, order=3):
    """Apply bandpass filter to remove noise outside breathing frequency range"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # Use second-order sections for numerical stability
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    filtered_data = signal.sosfilt(sos, data)

    return filtered_data


def load_and_preprocess_signals(participant_folder):
    """Load and preprocess signal data with filtering"""
    signals = {}

    # Define expected files and their sampling rates
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

                # Apply bandpass filter to respiratory signals
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


def load_events(participant_folder):
    """Load breathing event annotations"""
    events_file = os.path.join(participant_folder, 'events.csv')
    events = []

    if os.path.exists(events_file):
        try:
            df = pd.read_csv(events_file)
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])

            # Filter for relevant event types
            relevant_events = ['Hypopnea', 'Obstructive Apnea']
            df = df[df['event_type'].isin(relevant_events)]

            events = df.to_dict('records')
            print(f"Loaded {len(events)} relevant events")

        except Exception as e:
            print(f"Error loading events: {e}")

    return events


def create_windows(signals, window_size_sec=30, overlap_ratio=0.5):
    """Create overlapping windows from continuous signals"""

    # Find common time range
    start_times = [df.index.min() for df in signals.values()]
    end_times = [df.index.max() for df in signals.values()]

    common_start = max(start_times)
    common_end = min(end_times)

    # Calculate step size
    step_size_sec = window_size_sec * (1 - overlap_ratio)

    # Generate window timestamps
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


def extract_window_data(signals, window):
    """Extract signal data for a specific window"""
    window_data = {}

    for signal_name, df in signals.items():
        # Get data within window timeframe
        mask = (df.index >= window['start_time']) & (df.index < window['end_time'])
        window_signal = df.loc[mask, 'value'].values

        # Handle different sampling rates by resampling
        if signal_name == 'spo2':
            target_samples = int(30 * 4)  # 4 Hz for 30 seconds
        else:
            target_samples = int(30 * 32)  # 32 Hz for 30 seconds

        # Resample to target length if needed
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


def assign_labels(windows, events):
    """Assign labels to windows based on event overlap"""
    labeled_windows = []

    for window in windows:
        window_duration = (window['end_time'] - window['start_time']).total_seconds()
        best_label = 'Normal'
        max_overlap_ratio = 0

        for event in events:
            # Calculate overlap
            overlap_start = max(window['start_time'], event['start_time'])
            overlap_end = min(window['end_time'], event['end_time'])

            if overlap_start < overlap_end:
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                overlap_ratio = overlap_duration / window_duration

                # If overlap is > 50%, assign event label
                if overlap_ratio > 0.5 and overlap_ratio > max_overlap_ratio:
                    best_label = event['event_type']
                    max_overlap_ratio = overlap_ratio

        window['label'] = best_label
        labeled_windows.append(window)

    # Print label distribution
    label_counts = {}
    for window in labeled_windows:
        label = window['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"Label distribution: {label_counts}")
    return labeled_windows


def create_feature_matrix(signals, labeled_windows):
    """Create feature matrix from windowed signals"""

    features = []
    labels = []
    window_info = []

    for window in labeled_windows:
        try:
            # Extract window data
            window_data = extract_window_data(signals, window)

            # Combine all signals into feature vector
            feature_vector = []

            # Add nasal airflow (32 Hz * 30 sec = 960 samples)
            if 'nasal_airflow' in window_data:
                feature_vector.extend(window_data['nasal_airflow'])
            else:
                feature_vector.extend(np.zeros(960))

            # Add thoracic movement (32 Hz * 30 sec = 960 samples)
            if 'thoracic_movement' in window_data:
                feature_vector.extend(window_data['thoracic_movement'])
            else:
                feature_vector.extend(np.zeros(960))

            # Add SpO2 (4 Hz * 30 sec = 120 samples)
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


def process_participant(participant_folder, participant_id):
    """Process a single participant's data"""
    print(f"\nProcessing participant: {participant_id}")

    # Load and preprocess signals
    signals = load_and_preprocess_signals(participant_folder)
    if not signals:
        print(f"No signals found for participant {participant_id}")
        return None

    # Load events
    events = load_events(participant_folder)

    # Create windows
    windows = create_windows(signals)

    # Assign labels
    labeled_windows = assign_labels(windows, events)

    # Create feature matrix
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
            'nasal_airflow_samples': 960,  # 32Hz * 30sec
            'thoracic_movement_samples': 960,  # 32Hz * 30sec
            'spo2_samples': 120,  # 4Hz * 30sec
            'total_features': 2040  # 960 + 960 + 120
        }
    }


def save_dataset(dataset, output_dir, format='parquet'):
    """Save dataset in specified format"""
    os.makedirs(output_dir, exist_ok=True)

    if format == 'parquet':
        # Convert to DataFrame for Parquet
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
        parquet_path = os.path.join(output_dir, 'sleep_breathing_dataset.parquet')
        df.to_parquet(parquet_path, index=False)
        print(f"Dataset saved as Parquet: {parquet_path}")

        # Save metadata
        metadata = {
            'total_samples': len(df),
            'participants': [p['participant_id'] for p in dataset],
            'feature_info': dataset[0]['signal_info'] if dataset else {},
            'label_distribution': df['label'].value_counts().to_dict(),
            'creation_timestamp': pd.Timestamp.now().isoformat()
        }

        metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata saved: {metadata_path}")

    elif format == 'pickle':
        # Save as Pickle
        pickle_path = os.path.join(output_dir, 'sleep_breathing_dataset.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved as Pickle: {pickle_path}")

    elif format == 'csv':
        # Save as CSV (similar to Parquet but CSV format)
        all_features = []
        all_labels = []
        all_participants = []

        for participant_data in dataset:
            n_samples = len(participant_data['features'])
            all_features.extend(participant_data['features'].tolist())
            all_labels.extend(participant_data['labels'].tolist())
            all_participants.extend([participant_data['participant_id']] * n_samples)

        df = pd.DataFrame(all_features)
        df.columns = [f'feature_{i}' for i in range(df.shape[1])]
        df['label'] = all_labels
        df['participant_id'] = all_participants

        csv_path = os.path.join(output_dir, 'sleep_breathing_dataset.csv')
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved as CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Create labeled dataset from sleep data')
    parser.add_argument('-in_dir', '--input_dir', required=True,
                        help='Input directory containing participant folders')
    parser.add_argument('-out_dir', '--output_dir', required=True,
                        help='Output directory for dataset')
    parser.add_argument('--format', choices=['parquet', 'pickle', 'csv'],
                        default='parquet', help='Output format (default: parquet)')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    print(f"Creating dataset from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output format: {args.format}")

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
        participant_data = process_participant(participant_folder, participant_id)
        if participant_data:
            dataset.append(participant_data)

    if not dataset:
        print("No valid participant data found!")
        return

    print(f"\nDataset creation complete!")
    print(f"Total participants: {len(dataset)}")
    total_samples = sum(len(p['features']) for p in dataset)
    print(f"Total samples: {total_samples}")

    # Save dataset
    save_dataset(dataset, output_dir, args.format)
    print(f"Dataset saved successfully!")


if __name__ == "__main__":
    main()
