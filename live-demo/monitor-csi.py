import time
import os
import tensorflow as tf
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import re
import argparse

class CSIFileHandler(FileSystemEventHandler):
    def __init__(self, model, input_file, output_file):
        self.model = model
        self.input_file = input_file
        self.output_file = output_file
        
    def extract_csi_data(self, line):
        """Extract CSI data array from a line of text"""
        match = re.search(r'\[(.*?)\]', line)
        if match:
            # Convert string of numbers to numpy array
            csi_str = match.group(1)  # Get content without brackets
            try:
                # Split string by commas and convert to float numbers
                csi_values = [float(x.strip()) for x in csi_str.split(',')]
                return np.array(csi_values)
            except ValueError:
                return None
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
                csi_array = self.extract_csi_data(line)
                if csi_array is not None:
                    csi_data.append(csi_array)
            
            if not csi_data:
                print("No valid CSI data found")
                return None
                
            # Stack and reshape the data
            csi_data = np.stack(csi_data)
            csi_reshaped = csi_data.reshape(-1, 128, 50, 1)
            
            # Get predictions
            predictions = self.model.predict(csi_reshaped, verbose=0)
            if isinstance(predictions, tuple):
                presence_pred, location_pred = predictions
            else:
                # If model has single output
                presence_pred = predictions
                location_pred = None
            
            # Calculate average predictions
            avg_presence = int(np.mean(presence_pred) > 0.5)  # Convert to 0 or 1
            
            if location_pred is not None:
                avg_location = np.mean(location_pred, axis=0)
                return avg_presence, avg_location[0], avg_location[1]
            else:
                return avg_presence, None, None
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return None
    
    def save_prediction(self, prediction):
        """Save prediction to output file"""
        if prediction:
            state, x, y = prediction
            with open(self.output_file, 'w') as f:
                if x is not None and y is not None:
                    f.write(f"{state} {x:.0f} {y:.0f}\n")
                else:
                    f.write(f"{state}\n")
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.src_path == self.input_file:
            print("File update detected, processing...")
            prediction = self.process_txt()
            self.save_prediction(prediction)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Monitor CSI data and make predictions')
    parser.add_argument('--weights', type=str, required=True,
                      help='Path to model weights file')
    parser.add_argument('--input', type=str, default="live-data.txt",
                      help='Input file to monitor (default: live-data.txt)')
    parser.add_argument('--output', type=str, default="live-predictions.txt",
                      help='Output file for predictions (default: live-predictions.txt)')
    return parser.parse_args()

def start_monitoring(weights_path, input_file, output_file):
    """Start monitoring text file for updates"""
    # Load model
    print("Loading model...")
    try:
        model = tf.keras.models.load_model(weights_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
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
    args = parse_arguments()
    start_monitoring(args.weights, args.input, args.output)