import numpy as np
import time
import random

# Function to simulate receiving CSI data
def collect_csi_data():
    # Simulating CSI data in the required format
    data = {
        "timestamp": time.time(),
        "type": "Data",
        "role": "AP",
        "mac": "00:11:22:33:44:55",
        "rssi": random.randint(-100, -30),  # Simulated RSSI value
        "rate": random.choice([72, 144, 300, 600]),  # Simulated data rate
        "sig_mode": random.choice([1, 2]),  # Signal modes (just an example)
        "bandwidth": 20,  # Assuming a 20MHz bandwidth
        "real": [random.uniform(-1, 1) for _ in range(30)],  # Simulating 30 subcarriers
        "imag": [random.uniform(-1, 1) for _ in range(30)]   # Simulating 30 subcarriers
    }
    
    return data

# Function to save data to .npy
def save_data_to_npy(data, npy_filename="csi_data.npy"):
    # Extracting relevant data
    timestamp = data["timestamp"]
    rssi = data["rssi"]
    rate = data["rate"]
    sig_mode = data["sig_mode"]
    bandwidth = data["bandwidth"]
    real = np.array(data["real"])
    imag = np.array(data["imag"])

    # Creating an entry to save as a dictionary with all necessary data
    data_entry = {
        "timestamp": timestamp,
        "rssi": rssi,
        "rate": rate,
        "sig_mode": sig_mode,
        "bandwidth": bandwidth,
        "real": real,
        "imag": imag
    }

    # If the file doesn't exist, we create a new array; otherwise, append the data.
    try:
        # Load existing data (if any)
        existing_data = np.load(npy_filename, allow_pickle=True).item()
    except:
        existing_data = {"timestamps": [], "rssi": [], "rate": [], "sig_mode": [], "bandwidth": [], "real": [], "imag": []}
    
    # Append new data to the existing arrays
    existing_data["timestamps"].append(data_entry["timestamp"])
    existing_data["rssi"].append(data_entry["rssi"])
    existing_data["rate"].append(data_entry["rate"])
    existing_data["sig_mode"].append(data_entry["sig_mode"])
    existing_data["bandwidth"].append(data_entry["bandwidth"])
    existing_data["real"].append(data_entry["real"])
    existing_data["imag"].append(data_entry["imag"])

    # Save the updated data back to .npy file
    np.save(npy_filename, existing_data)

# Simulate data collection and saving every 1 second
for _ in range(10):  # Simulate 10 data entries (can be increased)
    csi_data = collect_csi_data()
    save_data_to_npy(csi_data, "csi_data.npy")
    time.sleep(1)  # Simulate a 1-second interval between data collection

print("CSI data has been collected and saved to csi_data.npy.")
