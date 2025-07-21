# Health Sensing Data Visualization

This repository contains a Python script designed to load, parse, and visualize physiological data collected during a sleep study for subject AP20. The script processes various CSV files containing time-series health sensing data, including airflow, SpO2 levels, respiratory efforts, sleep stages, and detected flow events.
![sleep_study_AP20_visualization_corrected](https://github.com/Quantamaster/Health-Sensing/blob/e064a11340dd707f92e414f45eb92e4faa53736d/sleep_study_AP20_visualization_corrected.png)

## Features

* **Multi-File Processing:** Automatically loads and processes multiple CSV files related to a single sleep study session.
* **Data Parsing:** Handles various data formats, including semicolon-separated values and different metadata structures (e.g., skipping header rows).
* **Time-Series Analysis:** Converts string timestamps into datetime objects for accurate time-series plotting.
* **Visualization:** Generates dedicated plots for:
    * Sleep Flow Events (e.g., Hypopnea)
    * Sleep Profile (Sleep Stages)
    * SpO2 (Blood Oxygen Saturation)
    * Thoracic Respiration
    * Airflow
DATASET : https://drive.google.com/drive/folders/1J95cTl574LLdj4uelYwjyv0094d8sOpD?usp=sharing

DeepMedico™ Sleep Breathing Irregularity Detection System
A complete end-to-end pipeline for detecting breathing irregularities (e.g., Apnea, Hypopnea) and classifying sleep stages from overnight sleep study signals using deep learning.

Features
Signal Visualization: Multi-signal PDF plots with overlaid event annotations.

Signal Preprocessing: Advanced bandpass filtering for breathing frequency extraction.

Dataset Engineering: 30-second, 50% overlapped windows with participant-wise labeling.

Efficient Storage: Dataset saved in Parquet format (ML-native, compressed).

Deep Learning Models: 1D CNN and Conv-LSTM architectures for robust time series classification.

Cross-Validation: Leave-One-Participant-Out CV to prevent data leakage.

Bonus Task: Sleep stage classification using the same framework.

Highly Modular: Each step can be run independently.

Directory Structure
text
DeepMedico/
├── Data/
│   ├── AP20/   # Example participant folder
│   │   ├── nasal_airflow.csv
│   │   ├── thoracic_movement.csv
│   │   ├── spo2.csv
│   │   ├── events.csv
│   │   └── sleep_profile.csv
│   └── ... (other participants)
├── Visualizations/
├── Dataset/
├── SleepStageDataset/
├── Results/
├── vis.py
├── create_dataset.py
├── modeling.py
├── sleep_stage_classification.py
├── requirements.txt
└── setup.py
Pipeline Overview
Visualize Signals

Plot and export comprehensive signal + annotation PDFs for QC/EDA.

Usage:

text
python vis.py -name "Data/AP20"
Create Dataset

Cleans and filters the signals.

Segments into 30s windows with 50% overlap.

Labels windows according to breathing event overlap.

Usage:

text
python create_dataset.py -in_dir "Data" -out_dir "Dataset" --format parquet
Train & Evaluate Models

1D CNN and Conv-LSTM, evaluated with leave-one-participant-out CV.

Per-class/classification metrics and mean/std result tables.
complete execution  pipeline:
# Step 1: Generate visualizations for each participant
python vis.py -name "Data/AP20"
python vis.py -name "Data/AP21" 
python vis.py -name "Data/AP22"
python vis.py -name "Data/AP23"
python vis.py -name "Data/AP24"

# Step 2: Create the dataset with preprocessing and windowing
python create_dataset.py -in_dir "Data" -out_dir "Dataset" --format parquet

# Step 3: Train and evaluate models with cross-validation
python modeling.py --dataset "Dataset/sleep_breathing_dataset.parquet" --model both --epochs 100

# Step 4: Bonus - Sleep stage classification
python sleep_stage_classification.py -in_dir "Data" -out_dir "SleepStageDataset" --train

Usage:

text
python modeling.py --dataset "Dataset/sleep_breathing_dataset.parquet" --model both --epochs 100
(Bonus) Sleep Stage Classification

Same pipeline as above, but with sleep stage rather than breathing event labels.

Usage:

text
python sleep_stage_classification.py -in_dir "Data" -out_dir "SleepStageDataset" --train
Input Data Format
Each participant subfolder (e.g. AP20/) should contain:

nasal_airflow.csv (timestamp,value)

thoracic_movement.csv (timestamp,value)

spo2.csv (timestamp,value)

events.csv (start_time,end_time,event_type)

sleep_profile.csv (start_time,end_time,sleep_stage)

Timestamps must be in an unambiguous format (ideally ISO 8601).

Requirements
Python >= 3.8

See requirements.txt

Install requirements:

text
pip install -r requirements.txt
Or install as a package:

text
python setup.py install
Example Pipeline (All Steps)
bash
python vis.py -name "Data/AP20"
python vis.py -name "Data/AP21"

python create_dataset.py -in_dir "Data" -out_dir "Dataset" --format parquet

python modeling.py --dataset "Dataset/sleep_breathing_dataset.parquet" --model both --epochs 100

python sleep_stage_classification.py -in_dir "Data" -out_dir "SleepStageDataset" --train
Output
Visualizations/: per-participant signal PDF files (EDA).

Dataset/: Parquet file with windowed features and labels.

Results/: Model performance metrics (JSON and logs).

SleepStageDataset/: Sleep stage dataset, metadata, and (if --train) model performance.


Advanced Notes
Filtering: Bandpass 0.17-0.4 Hz (removes movement artifacts and drift).

Windowing: 30 seconds, 50% overlap, matching standard sleep study analysis.

Class Labels: 'Normal', 'Hypopnea', 'Obstructive Apnea' (event labeling).

Sleep Stages: 'Wake', 'N1', 'N2', 'N3', 'REM' (bonus/extension).

Evaluation: Only leave-one-subject-out prevents data leakage. Random splits are inappropriate for personalized physiological data.
![sleep monitor](https://github.com/Quantamaster/Health-Sensing/blob/47b84bfc9658fbab09b1379e1911104aadae83e2/sleep%20monitor.png)
