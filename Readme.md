# Health Sensing Data Visualization

This repository contains a Python script designed to load, parse, and visualize physiological data collected during a sleep study for subject AP20. The script processes various CSV files containing time-series health sensing data, including airflow, SpO2 levels, respiratory efforts, sleep stages, and detected flow events.

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
## Requirements

The script is written in Python and requires the following libraries:

* Python 3.x
* Pandas
* Matplotlib

## Installation

1.  **Clone the repository or save the Python script:** Ensure the Python script (`vis.py` based on the context) and the data are in the correct directory structure.
2.  **Install dependencies:** Use pip to install the required libraries:

```bash
pip install pandas matplotlib
