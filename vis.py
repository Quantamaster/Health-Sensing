import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import signal
import argparse

def low_pass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)

def parse_timestamp(ts):
    ts = ts.replace(',', '.')
    return pd.to_datetime(ts, format='%d.%m.%Y %H:%M:%S.%f')

def read_signal(file_path):
    df = pd.read_csv(file_path, sep=';', names=['timestamp', 'value'])
    df['timestamp'] = df['timestamp'].apply(parse_timestamp)
    df.set_index('timestamp', inplace=True)
    return df

def read_events(file_path):
    events = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 3:
                time_range, _, event_type, _ = parts
                start_str, end_str = time_range.split('-')
                start = parse_timestamp(start_str)
                end = parse_timestamp(f"{start.date().strftime('%d.%m.%Y')} {end_str.replace(',', '.')}")
                events.append((start, end, event_type))
    return events

def plot_signals(airflow, thoracic, spo2, events, participant_id):
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(airflow.index, airflow['filtered'], label='Nasal Airflow')
    axes[1].plot(thoracic.index, thoracic['filtered'], label='Thoracic Movement')
    axes[2].plot(spo2.index, spo2['filtered'], label='SpO₂')

    color_map = {'Hypopnea': 'blue', 'Obstructive Apnea': 'red'}
    for ax in axes:
        for start, end, event_type in events:
            if event_type in color_map:
                ax.axvspan(start, end, color=color_map[event_type], alpha=0.3)

    legend_elements = [Patch(facecolor='blue', alpha=0.3, label='Hypopnea'),
                       Patch(facecolor='red', alpha=0.3, label='Obstructive Apnea')]
    axes[0].legend(handles=legend_elements)

    axes[0].set_title('Nasal Airflow')
    axes[1].set_title('Thoracic Movement')
    axes[2].set_title('SpO₂')
    fig.suptitle(f"Sleep Data for {participant_id}")
    plt.tight_layout()
    os.makedirs('Visualizations', exist_ok=True)
    plt.savefig(f'Visualizations/{participant_id}.pdf')
    plt.close()

def main(input_dir):
    participant_id = os.path.basename(input_dir)
    airflow_file = os.path.join(input_dir, 'Flow - 30-05-2024.csv')
    thoracic_file = os.path.join(input_dir, 'Thorac - 30-05-2024.csv')
    spo2_file = os.path.join(input_dir, 'SPO2 - 30-05-2024.csv')
    events_file = os.path.join(input_dir, 'Flow Events - 30-05-2024.csv')

    airflow = read_signal(airflow_file)
    thoracic = read_signal(thoracic_file)
    spo2 = read_signal(spo2_file)
    events = read_events(events_file)

    # Apply filter
    airflow['filtered'] = low_pass_filter(airflow['value'].values, cutoff=1, fs=32)
    thoracic['filtered'] = low_pass_filter(thoracic['value'].values, cutoff=1, fs=32)
    spo2['filtered'] = low_pass_filter(spo2['value'].values, cutoff=1, fs=4)

    plot_signals(airflow, thoracic, spo2, events, participant_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', required=True, help='Path to participant data folder')
    args = parser.parse_args()
    main(args.name)