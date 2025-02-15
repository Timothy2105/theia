import time
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils import ESP32CSIMultiTaskModel, parse_csi_data

class CSIFileHandler(FileSystemEventHandler):
    def __init__(self, model, input_file, output_file):
        self.model = model
        self.input_file = input_file
        self.output_file = output_file
        
    def process_csv(self):
        """Process CSV file and return predictions"""
        try:
            # Read CSV
            df = pd.read_csv(self.input_file)
            
            # Parse CSI data
            csi_data = np.stack(df['CSI_DATA'].apply(parse_csi_data).values)
            
            # Reshape for model input (batch_size, n_subcarriers, window_size, channels)
            csi_reshaped = csi_data.reshape(-1, 128, 50, 1)
            
            # Get predictions
            presence_pred, location_pred = self.model.model.predict(csi_reshaped, verbose=0)
            
            # Calculate average predictions
            avg_presence = int(np.mean(presence_pred) > 0.5)  # Convert to 0 or 1
            avg_location = np.mean(location_pred, axis=0)
            
            return avg_presence, avg_location[0], avg_location[1]
            
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")
            return None
    
    def save_prediction(self, prediction):
        """Save prediction to output file"""
        if prediction:
            state, x, y = prediction
            with open(self.output_file, 'w') as f:
                f.write(f"{state} {x:.0f} {y:.0f}\n")
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.src_path == self.input_file:
            print("CSV update detected, processing...")
            prediction = self.process_csv()
            self.save_prediction(prediction)

def start_monitoring(input_file, output_file):
    """Start monitoring CSV file for updates"""
    # Load model
    print("Loading model...")
    model = ESP32CSIMultiTaskModel()
    model.model = tf.keras.models.load_model('../model/primitive_weights.h5')
    print("Model loaded successfully")
    
    # Create event handler and observer
    event_handler = CSIFileHandler(model, input_file, output_file)
    observer = Observer()
    observer.schedule(event_handler, os.path.dirname(input_file), recursive=False)
    
    # Start monitoring
    print(f"Starting to monitor file: {input_file}")
    print(f"Saving predictions to: {output_file}")
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Monitoring stopped")
    
    observer.join()

if __name__ == "__main__":
    INPUT_FILE = "live-data.csv"
    OUTPUT_FILE = "live-predictions.txt"
    
    start_monitoring(INPUT_FILE, OUTPUT_FILE)