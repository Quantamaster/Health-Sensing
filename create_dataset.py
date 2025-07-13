import os
import pandas as pd
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import argparse

# (Include the low_pass_filter and parse_timestamp functions from vis.py)

def resample_signal(df, target_freq, time_index):
    times = (df.index - df.index[0]).total_seconds().values
    f = interp1d(times, df['value'].values, kind='linear', fill_value="extrapolate")
    target_times = (time_index - time_index[0]).total_seconds().values
    resampled_values = f(target_times)
    return resampled_values

def create_windows(data, window_size, step_size):
    num_samples = data.shape[0]
    windows = []
    for start in range(0, num_samples - window_size + 1, step_size):
        window = data[start:start + window_size]
        windows.append(window)
    return np.array(windows)

def label_window(window_start, window_end, events):
    overlap_threshold = pd.Timedelta(seconds=15)
    for event_start, event_end, event_type in events:
        if event_type in ['Hypopnea', 'Obstructive Apnea']:
            overlap = min(window_end, event_end) - max(window_start, event_start)
            if overlap > overlap_threshold:
                return event_type
    return 'Normal'

def process_participant(input_dir, output_dir, participant_id):
    # Read and filter signals (similar to vis.py)
    # ...

    # Determine common time index
    start_time = min(airflow.index.min(), thoracic.index.min(), spo2.index.min())
    end_time = max(airflow.index.max(), thoracic.index.max(), spo2.index.max())
    time_index = pd.date_range(start=start_time, end=end_time, freq='31.25ms')

    # Resample signals to 32 Hz
    airflow_resampled = resample_signal(airflow, 32, time_index)
    thoracic_resampled = resample_signal(thoracic, 32, time_index)
    spo2_resampled = resample_signal(spo2, 4, time_index)  # Assuming spo2 is already at 4 Hz, but resampled to 32 Hz

    # Stack signals
    data = np.stack([airflow_resampled, thoracic_resampled, spo2_resampled], axis=-1)

    # Create windows
    window_size = 30 * 32  # 30 seconds * 32 Hz
    step_size = 15 * 32  # 15 seconds * 32 Hz
    windows = create_windows(data, window_size, step_size)

    # Label windows
    labels = []
    for i in range(windows.shape[0]):
        window_start = time_index[i * step_size]
        window_end = window_start + pd.Timedelta(seconds=30)
        label = label_window(window_start, window_end, events)
        labels.append(label)

    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'{participant_id}_data.npy'), windows)
    np.save(os.path.join(output_dir, f'{participant_id}_labels.npy'), np.array(labels))

def main(in_dir, out_dir):
    for participant_folder in os.listdir(in_dir):
        participant_id = participant_folder
        input_path = os.path.join(in_dir, participant_folder)
        process_participant(input_path, out_dir, participant_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', required=True, help='Input directory containing participant folders')
    parser.add_argument('-out_dir', required=True, help='Output directory for dataset')
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)