"""
DeepMedico Sleep Data Visualization Script
Generates comprehensive PDF visualizations for participant sleep data
Usage: python vis.py -name "Data/AP20"
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def load_signal_data(participant_folder):
    """Load and process signal data for a participant"""
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
                # Assume timestamp column is first, signal value is second
                df.columns = ['timestamp', 'value']
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                signals[signal_name] = df
                print(f"Loaded {signal_name}: {len(df)} samples")
            except Exception as e:
                print(f"Error loading {signal_name}: {e}")
        else:
            print(f"File not found: {file_path}")

    return signals


def load_events_data(participant_folder):
    """Load breathing event annotations"""
    events_file = os.path.join(participant_folder, 'events.csv')
    events = []

    if os.path.exists(events_file):
        try:
            df = pd.read_csv(events_file)
            # Expected columns: start_time, end_time, event_type
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])
            events = df.to_dict('records')
            print(f"Loaded {len(events)} events")
        except Exception as e:
            print(f"Error loading events: {e}")
    else:
        print(f"Events file not found: {events_file}")

    return events


def resample_signals_for_visualization(signals):
    """Resample all signals to common timebase for visualization"""
    # Find common time range
    start_times = [df.index.min() for df in signals.values()]
    end_times = [df.index.max() for df in signals.values()]

    common_start = max(start_times)
    common_end = min(end_times)

    # Create common time index (1Hz for visualization)
    common_index = pd.date_range(start=common_start, end=common_end, freq='1S')

    resampled = {}
    for signal_name, df in signals.items():
        # Resample to 1Hz using linear interpolation
        resampled_df = df.reindex(df.index.union(common_index)).interpolate().reindex(common_index)
        resampled[signal_name] = resampled_df

    return resampled, common_start, common_end


def create_visualization(signals, events, participant_name, output_dir):
    """Create comprehensive PDF visualization"""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Resample signals for visualization
    resampled_signals, start_time, end_time = resample_signals_for_visualization(signals)

    pdf_path = os.path.join(output_dir, f'{participant_name}_visualization.pdf')

    with PdfPages(pdf_path) as pdf:
        # Main visualization page
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle(f'Sleep Study Visualization - {participant_name}', fontsize=16, fontweight='bold')

        # Plot each signal
        signal_names = ['nasal_airflow', 'thoracic_movement', 'spo2']
        signal_labels = ['Nasal Airflow', 'Thoracic Movement', 'SpO₂ (%)']
        colors = ['blue', 'green', 'red']

        for i, (signal_name, label, color) in enumerate(zip(signal_names, signal_labels, colors)):
            ax = axes[i]

            if signal_name in resampled_signals:
                data = resampled_signals[signal_name]['value']
                ax.plot(data.index, data.values, color=color, linewidth=0.5, alpha=0.7)

                # Overlay events
                for event in events:
                    if event['start_time'] >= start_time and event['end_time'] <= end_time:
                        event_color = 'red' if 'Apnea' in event['event_type'] else 'orange'
                        alpha = 0.3
                        ax.axvspan(event['start_time'], event['end_time'],
                                   color=event_color, alpha=alpha, label=event['event_type'])

            ax.set_ylabel(label, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(start_time, end_time)

            # Format x-axis
            if i == 2:  # Only show x-axis labels on bottom plot
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.set_xlabel('Time (HH:MM)', fontsize=12)
            else:
                ax.set_xticklabels([])

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()

        # Summary statistics page
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Signal Quality and Event Summary - {participant_name}', fontsize=16, fontweight='bold')

        # Signal statistics
        stats_data = []
        for signal_name in signal_names:
            if signal_name in resampled_signals:
                data = resampled_signals[signal_name]['value'].dropna()
                stats_data.append([
                    signal_name.replace('_', ' ').title(),
                    f"{data.mean():.2f}",
                    f"{data.std():.2f}",
                    f"{data.min():.2f}",
                    f"{data.max():.2f}",
                    f"{len(data)}"
                ])

        # Create statistics table
        ax1.axis('tight')
        ax1.axis('off')
        table = ax1.table(cellText=stats_data,
                          colLabels=['Signal', 'Mean', 'Std', 'Min', 'Max', 'Samples'],
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax1.set_title('Signal Statistics', fontsize=14, fontweight='bold')

        # Event distribution
        event_types = {}
        for event in events:
            event_type = event['event_type']
            if event_type in event_types:
                event_types[event_type] += 1
            else:
                event_types[event_type] = 1

        if event_types:
            ax2.bar(event_types.keys(), event_types.values(),
                    color=['red', 'orange', 'yellow'][:len(event_types)])
            ax2.set_title('Event Distribution', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Count')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        else:
            ax2.text(0.5, 0.5, 'No Events Found', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Event Distribution', fontsize=14, fontweight='bold')

        # Signal quality indicators
        quality_scores = []
        quality_labels = []
        for signal_name in signal_names:
            if signal_name in resampled_signals:
                data = resampled_signals[signal_name]['value'].dropna()
                # Simple quality score based on data completeness and variability
                completeness = len(data) / len(resampled_signals[signal_name])
                variability = data.std() / abs(data.mean()) if data.mean() != 0 else 0
                quality_score = min(100, (completeness * 50 + min(variability * 50, 50)))
                quality_scores.append(quality_score)
                quality_labels.append(signal_name.replace('_', ' ').title())

        ax3.barh(quality_labels, quality_scores, color=['blue', 'green', 'red'][:len(quality_scores)])
        ax3.set_title('Signal Quality Score', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Quality Score (%)')
        ax3.set_xlim(0, 100)

        # Duration and timing info
        duration_hours = (end_time - start_time).total_seconds() / 3600
        recording_info = [
            ['Recording Start', start_time.strftime('%Y-%m-%d %H:%M:%S')],
            ['Recording End', end_time.strftime('%Y-%m-%d %H:%M:%S')],
            ['Total Duration', f'{duration_hours:.2f} hours'],
            ['Total Events', str(len(events))],
            ['Events per Hour', f'{len(events) / duration_hours:.1f}' if duration_hours > 0 else 'N/A']
        ]

        ax4.axis('tight')
        ax4.axis('off')
        table2 = ax4.table(cellText=recording_info,
                           colLabels=['Parameter', 'Value'],
                           cellLoc='left',
                           loc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1, 1.5)
        ax4.set_title('Recording Information', fontsize=14, fontweight='bold')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()

    print(f"Visualization saved to: {pdf_path}")
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description='Generate sleep study visualizations')
    parser.add_argument('-name', '--name', required=True,
                        help='Participant folder path (e.g., "Data/AP20")')

    args = parser.parse_args()

    participant_folder = args.name
    participant_name = os.path.basename(participant_folder)
    output_dir = 'Visualizations'

    print(f"Processing participant: {participant_name}")
    print(f"Data folder: {participant_folder}")

    # Load data
    signals = load_signal_data(participant_folder)
    events = load_events_data(participant_folder)

    if not signals:
        print("No signal data found!")
        return

    # Create visualization
    pdf_path = create_visualization(signals, events, participant_name, output_dir)
    print(f"Visualization complete!")


if __name__ == "__main__":
    main()
