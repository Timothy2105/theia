import time
import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import argparse
import re

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.utils import ESP32CSIMultiTaskModel, parse_csi_data

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
            current_size = os.path.getsize(self.input_file)
            
            # Read all lines from the file
            with open(self.input_file, 'r') as f:
                all_lines = f.readlines()
                
            if not all_lines:
                print("No lines to process")
                return None
                
            print(f"\nProcessing {len(all_lines)} lines...")
            
            # Extract CSI data from each line
            csi_data = []
            for line in all_lines:
                csi_str = self.extract_csi_data(line)
                if csi_str:
                    try:
                        parsed_data = parse_csi_data(csi_str)
                        csi_data.append(parsed_data)
                        print("Successfully parsed CSI data from line")
                        print(f"Parsed data shape: {np.array(parsed_data).shape}")
                    except Exception as e:
                        print(f"Error parsing CSI data: {str(e)}")
                        print(f"Problematic line: {line}")
            
            if not csi_data:
                print("No valid CSI data found in lines")
                return None
                
            # Stack and reshape the data
            csi_data = np.stack(csi_data)
            print(f"CSI data shape before reshape: {csi_data.shape}")
            print(f"Total elements in array: {csi_data.size}")
            
            # Modify the reshaping logic to match the model's expected input shape (None, 128, 50, 1)
            if len(csi_data.shape) == 2:  # Shape is (num_samples, 128)
                num_samples = csi_data.shape[0]
                # First reshape to (num_samples, 128)
                intermediate = csi_data.reshape(num_samples, 128)
                # Then expand to (num_samples, 128, 50) by repeating values
                expanded = np.repeat(intermediate[:, :, np.newaxis], 50, axis=2)
                # Finally add channel dimension to get (num_samples, 128, 50, 1)
                csi_reshaped = expanded[..., np.newaxis]
            else:
                raise ValueError(f"Unexpected data shape: {csi_data.shape}. Expected (num_samples, 128)")
            
            print(f"CSI data shape after reshape: {csi_reshaped.shape}")
            
            # Get predictions
            presence_pred, location_pred = self.model.model.predict(csi_reshaped, verbose=0)
            
            # Print individual predictions
            print("\nIndividual predictions:")
            print("Frame\tPresence\tLocation (X, Y)")
            print("-" * 40)
            for i in range(len(presence_pred)):
                presence = "Yes" if presence_pred[i] > 0.5 else "No"
                print(f"{i+1}\t{presence}\t\t({location_pred[i][0]:.1f}, {location_pred[i][1]:.1f})")
            
            # Calculate and print average predictions
            avg_presence = int(np.mean(presence_pred) > 0.5)
            avg_location = np.mean(location_pred, axis=0)
            
            print("\nAveraged predictions:")
            print(f"Presence: {'Yes' if avg_presence else 'No'}")
            print(f"Location: (X: {avg_location[0]:.1f}, Y: {avg_location[1]:.1f})")
            print("-" * 40)
            
            return avg_presence, avg_location[0], avg_location[1]
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def save_prediction(self, prediction):
        """Save prediction to output file"""
        if prediction:
            state, x, y = prediction
            try:
                with open(self.output_file, 'w') as f:
                    f.write(f"{state} {x:.0f} {y:.0f}\n")
                    # Ensure the write is flushed to disk
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                print(f"Error saving prediction: {str(e)}")
    
    def on_modified(self, event):
        """Handle file modification events"""
        try:
            if not os.path.exists(self.input_file):
                print(f"Warning: Input file {self.input_file} does not exist")
                return
                
            # Process modifications to input file
            if event.src_path == os.path.abspath(self.input_file):
                print(f"\nFile modification detected: {event.src_path}")
                prediction = self.process_txt()
                if prediction:
                    self.save_prediction(prediction)
                    print("Prediction saved successfully")
            
        except Exception as e:
            print(f"Error in on_modified: {str(e)}")

def start_monitoring(input_file, output_file, weights_path):
    """Start monitoring text file for updates"""
    # Ensure input file exists
    if not os.path.exists(input_file):
        open(input_file, 'a').close()
        print(f"Created empty input file: {input_file}")
    
    # Get absolute paths
    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)
    weights_path = os.path.abspath(weights_path)
    
    # Verify weights file exists
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found at: {weights_path}")
    
    # Load model
    print(f"Loading model from: {weights_path}")
    model = ESP32CSIMultiTaskModel()
    model.model = tf.keras.models.load_model(weights_path)
    print("Model loaded successfully")
    
    # Create event handler and observer
    event_handler = CSIFileHandler(model, input_file, output_file)
    observer = Observer()
    
    # Schedule monitoring for the directory containing the input file
    watch_dir = os.path.dirname(input_file)
    print(f"Setting up watchdog observer for directory: {watch_dir}")
    observer.schedule(event_handler, watch_dir, recursive=False)
    
    # Start monitoring
    print(f"Starting to monitor file: {input_file}")
    print(f"Saving predictions to: {output_file}")
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nMonitoring stopped")
    
    observer.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor CSI data and make predictions')
    parser.add_argument('--input', default="live-data.txt", help='Input file to monitor')
    parser.add_argument('--output', default="live-predictions.txt", help='Output file for predictions')
    parser.add_argument('--weights', required=True, help='Path to model weights file (best_weights.h5)')
    
    args = parser.parse_args()
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Weights file: {args.weights}")
    
    start_monitoring(args.input, args.output, args.weights)