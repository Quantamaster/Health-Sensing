import pandas as pd
import os

# Define the file paths
file_paths = [
    "Flow Events - 30-05-2024.txt",
    "Sleep profile - 30-05-2024.txt",
    "SPO2 - 30-05-2024.txt",
    "Thorac - 30-05-2024.txt",
    "Flow - 30-05-2024.txt"
]

# Function to convert text file to CSV
def convert_txt_to_csv(input_file, output_file):
    # Read the text file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Process the lines to extract Data
    data = []
    for line in lines:
        # Assuming Data is separated by whitespace, adjust as needed
        row = line.strip().split()
        data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_file, index=False, header=False)

# Convert each text file to CSV
for file_path in file_paths:
    output_file = file_path.replace('.txt', '.csv')
    convert_txt_to_csv(file_path, output_file)
    print(f"Converted {file_path} to {output_file}")
