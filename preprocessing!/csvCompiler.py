import pandas as pd
import time
import random

# Define columns for CSV file
columns = ["state", "x", "y", "raw_data"]

# Initialize an empty dataframe
df = pd.DataFrame(columns=columns)

# Function to simulate receiving CSI data (now takes x, y, state, and simulated raw data)
def collect_csi_data(x_position, y_position, state, simulated_raw_data):
    data = {
        "state": state,  # Presence state (1 for present, 0 for absent)
        "x": x_position,  # x-coordinate (decimeters)
        "y": y_position,  # y-coordinate (decimeters)
        "raw_data": simulated_raw_data  # Simulated raw data
    }
    return data

# Function to save data to CSV
def save_data_to_csv(data, csv_filename="csi_data.csv"):
    global df
    # Append the data to the DataFrame
    df = df.append(data, ignore_index=True)
    # Write DataFrame to CSV
    df.to_csv(csv_filename, index=False)

# Data collection loop (you'll run this *for each* grid point)
while True:
    try:
        x_position = float(input("Enter x-coordinate (decimeters, 0-50, or 'done'): "))
        if x_position == "done":  # Allow an exit
            break
        y_position = float(input("Enter y-coordinate (decimeters, 0-50): "))
        state = int(input("Enter state (0 for absent, 1 for present): ")) # Always 1 when calibrating
        
        # Simulate raw data (replace with your *actual* ESP32 CSI readings)
        simulated_raw_data = [random.uniform(-1, 1) for _ in range(10)]  # Placeholder

        csi_data = collect_csi_data(x_position, y_position, state, simulated_raw_data)
        save_data_to_csv(csi_data, "csi_data.csv")

        print("Data saved. Enter next coordinate, or 'done'.")

    except ValueError:
        print("Invalid input. Please enter numbers.")

print("CSI data collection complete.  Check csi_data.csv")
