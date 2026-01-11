# Health Sensing Data Visualization
🧠 DeepMedico™: Health Sensing & Sleep Breathing Irregularity Detection
<p align="center"> <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/Deep%20Learning-1D%20CNN%20%7C%20Conv--LSTM-brightgreen"/> <img src="https://img.shields.io/badge/Domain-Sleep%20Health%20%7C%20Physiology-purple"/> <img src="https://img.shields.io/badge/Data-Time--Series%20Signals-orange"/> <img src="https://img.shields.io/badge/License-MIT-lightgrey"/> </p> <p align="center"> <b>An end-to-end deep learning pipeline for sleep breathing irregularity detection and sleep stage classification</b> </p>

This repository contains a Python script designed to load, parse, and visualize physiological data collected during a sleep study for subject AP20. The script processes various CSV files containing time-series health sensing data, including airflow, SpO2 levels, respiratory efforts, sleep stages, and detected flow events.
![sleep_study_AP20_visualization_corrected](https://github.com/Quantamaster/Health-Sensing/blob/e064a11340dd707f92e414f45eb92e4faa53736d/sleep_study_AP20_visualization_corrected.png)

## 📌 Table of Contents

- [Abstract](#-abstract)
- [Keywords](#-keywords)
- [Introduction](#-introduction)
- [Features](#features)
- [Dataset](#-dataset)
- [Pipeline Architecture](#-pipeline-architecture)
- [Methodology](#-methodology)
- [Directory Structure](#-directory-structure)
- [Usage](#-usage)
- [Input Format](#-input-format)
- [Requirements](#requirements)
- [Outputs](#-outputs)
- [Advanced Notes](#-advanced-notes)


## 🧾 Abstract

Sleep-related breathing disorders such as Obstructive Sleep Apnea (OSA) and Hypopnea require accurate and scalable detection systems.
DeepMedico™ presents a research-grade, end-to-end framework that integrates:

Multi-modal physiological signal visualization

Signal preprocessing and dataset engineering

Deep temporal modeling using CNN and Conv-LSTM

Subject-independent evaluation using Leave-One-Participant-Out CV

The pipeline is designed to be modular, reproducible, and clinically relevant.

## 🔑 Keywords

Sleep Study · Apnea · Hypopnea · Physiological Signals · Time-Series · CNN · Conv-LSTM · SpO₂ · Respiration · Deep Learning

## 🧠 Introduction

Manual polysomnography (PSG) analysis is costly and time-consuming. Automated systems must handle:

Inter-subject variability

Temporal dependencies

Strict evaluation protocols to prevent data leakage

DeepMedico™ addresses these challenges with a fully automated and explainable pipeline, spanning visualization to model evaluation.

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
 
## 📊 Dataset

Each participant’s overnight recording includes:

Nasal airflow

Thoracic respiratory movement

SpO₂ levels

Expert-annotated breathing events

Sleep stage labels
DATASET : https://drive.google.com/drive/folders/1J95cTl574LLdj4uelYwjyv0094d8sOpD?usp=sharing

DeepMedico™ Sleep Breathing Irregularity Detection System
A complete end-to-end pipeline for detecting breathing irregularities (e.g., Apnea, Hypopnea) and classifying sleep stages from overnight sleep study signals using deep learning.

## 🏗 Pipeline Architecture

Raw Signals
   ↓
Visualization (EDA & QC)
   ↓
Preprocessing & Filtering
   ↓
Windowing & Labeling
   ↓
Parquet Dataset
   ↓
CNN / Conv-LSTM Models
   ↓
LOPO Cross-Validation

## 🧪 Methodology
<details> <summary><b>📈 Signal Visualization</b></summary>

Multi-signal time-aligned plots

Apnea/Hypopnea overlays

Exported as per-participant PDFs for EDA and QC

</details> <details> <summary><b>🧹 Signal Preprocessing</b></summary>

Bandpass filtering (0.17–0.4 Hz)

Timestamp normalization

Noise and drift suppression

</details> <details> <summary><b>📦 Dataset Engineering</b></summary>

30-second windows

50% overlap

Event-based labeling

Parquet storage for efficiency

</details> <details> <summary><b>🤖 Deep Learning Models</b></summary>

1D CNN – local temporal features

Conv-LSTM – long-range dependencies

Multi-class classification

</details> <details> <summary><b>📏 Evaluation Protocol</b></summary>

Leave-One-Participant-Out CV

Per-class Precision / Recall / F1

Mean ± Std across folds

</details>

Bonus Task: Sleep stage classification using the same framework.

Highly Modular: Each step can be run independently.

## 📁 Directory Structure

DeepMedico/
├── Data/
│   ├── AP20/                      # Example participant folder
│   │   ├── nasal_airflow.csv      # Nasal airflow signal (timestamp, value)
│   │   ├── thoracic_movement.csv  # Thoracic respiration signal
│   │   ├── spo2.csv               # Blood oxygen saturation (SpO₂)
│   │   ├── events.csv             # Apnea/Hypopnea annotations
│   │   └── sleep_profile.csv      # Sleep stage labels
│   └── ...                        # Other participant folders (AP21, AP22, etc.)
│
├── Visualizations/                # Generated signal + annotation PDFs
├── Dataset/                       # Windowed breathing-event dataset (Parquet)
├── SleepStageDataset/             # Sleep stage dataset & features
├── Results/                       # Model metrics, logs, CV results
│
├── vis.py                         # Signal visualization & EDA
├── create_dataset.py              # Preprocessing, windowing, labeling
├── modeling.py                    # 1D CNN & Conv-LSTM training + evaluation
├── sleep_stage_classification.py  # Bonus: sleep stage classification
│
├── requirements.txt               # Python dependencies
└── setup.py                       # Package installation

Plot and export comprehensive signal + annotation PDFs for QC/EDA.

## 🚀 Usage

python vis.py -name "Data/AP20"
Create Dataset

Cleans and filters the signals.

Segments into 30s windows with 50% overlap.

Labels windows according to breathing event overlap.

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

Snippets:
python modeling.py --dataset "Dataset/sleep_breathing_dataset.parquet" --model both --epochs 100
(Bonus) Sleep Stage Classification
Same pipeline as above, but with sleep stage rather than breathing event labels.
Snippets:
python sleep_stage_classification.py -in_dir "Data" -out_dir "SleepStageDataset" --train

## 📥 Input Format
Each participant subfolder (e.g. AP20/) should contain:

nasal_airflow.csv (timestamp,value)

thoracic_movement.csv (timestamp,value)

spo2.csv (timestamp,value)

events.csv (start_time,end_time,event_type)

sleep_profile.csv (start_time,end_time,sleep_stage)

Timestamps must be in an unambiguous format (ideally ISO 8601).

## Requirements
Python >= 3.8

See requirements.txt

Install requirements:

pip install -r requirements.txt
Or install as a package:


python setup.py install
Example Pipeline (All Steps)
bash
python vis.py -name "Data/AP20"
python vis.py -name "Data/AP21"

python create_dataset.py -in_dir "Data" -out_dir "Dataset" --format parquet

python modeling.py --dataset "Dataset/sleep_breathing_dataset.parquet" --model both --epochs 100

python sleep_stage_classification.py -in_dir "Data" -out_dir "SleepStageDataset" --train

## 📤 Outputs
Visualizations/: per-participant signal PDF files (EDA).

Dataset/: Parquet file with windowed features and labels.

Results/: Model performance metrics (JSON and logs).

SleepStageDataset/: Sleep stage dataset, metadata, and (if --train) model performance.


## 🧠 Advanced Notes
Filtering: Bandpass 0.17-0.4 Hz (removes movement artifacts and drift).

Windowing: 30 seconds, 50% overlap, matching standard sleep study analysis.

Class Labels: 'Normal', 'Hypopnea', 'Obstructive Apnea' (event labeling).

Sleep Stages: 'Wake', 'N1', 'N2', 'N3', 'REM' (bonus/extension).

Evaluation: Only leave-one-subject-out prevents data leakage. Random splits are inappropriate for personalized physiological data.
![sleep monitor](https://github.com/Quantamaster/Health-Sensing/blob/47b84bfc9658fbab09b1379e1911104aadae83e2/sleep%20monitor.png)





