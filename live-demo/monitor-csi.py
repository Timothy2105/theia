import time
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils import ESP32CSIMultiTaskModel, parse_csi_data
import re

class CSIFileHandler(FileSystemEventHandler):
    def __init__(self, model, input_file, output_file):
        self.model = model
        self.input_file = input_file
        self.output_file = output_file
        
    def extract_csi_data(self, line):
        """Extract CSI data array from a line of text"""
        # Find content between square brackets
        match = re.search(r'\[(.*?)\]', line)
        if match:
            return match.group(0)  # Return the full bracket content
        return None
        
    def process_txt(self):
        """Process text file and return predictions"""
        try:
            # Read all lines from the file
            with open(self.input_file, 'r') as f:
                lines = f.readlines()
            
            # Extract CSI data from each line
            csi_data = []
            for line in lines:
                csi_str = self.extract_csi_data(line)
                if csi_str:
                    csi_data.append(parse_csi_data(csi_str))
            
            if not csi_data:
                print("No valid CSI data found")
                return None
                
            # Stack and reshape the data
            csi_data = np.stack(csi_data)
            csi_reshaped = csi_data.reshape(-1, 128, 50, 1)
            
            # Get predictions
            presence_pred, location_pred = self.model.model.predict(csi_reshaped, verbose=0)
            
            # Calculate average predictions
            avg_presence = int(np.mean(presence_pred) > 0.5)  # Convert to 0 or 1
            avg_location = np.mean(location_pred, axis=0)
            
            return avg_presence, avg_location[0], avg_location[1]
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
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
            print("File update detected, processing...")
            prediction = self.process_txt()
            self.save_prediction(prediction)

def start_monitoring(input_file, output_file):
    """Start monitoring text file for updates"""
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
    INPUT_FILE = "live-data.txt"
    OUTPUT_FILE = "live-predictions.txt"
    
    start_monitoring(INPUT_FILE, OUTPUT_FILE)