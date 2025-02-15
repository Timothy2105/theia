import pandas as pd
import os

def merge_data():
    """Merge all processed CSV files into one dataset"""
    # Create merged-data directory if it doesn't exist
    os.makedirs('merged-data', exist_ok=True)
    
    labelled_dir = 'labelled-data'
    csv_files = [f for f in os.listdir(labelled_dir) if f.endswith('.csv')]
    
    print(f"\nFound {len(csv_files)} processed files to merge:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file}")
    
    all_data = []
    
    for i, csv_file in enumerate(csv_files, 1):
        file_path = os.path.join(labelled_dir, csv_file)
        print(f"\nReading file {i}/{len(csv_files)}: {csv_file}")
        
        df = pd.read_csv(file_path)
        all_data.append(df)
    
    merged_data = pd.concat(all_data, ignore_index=True)
    
    output_file = os.path.join('merged-data', "merged_labelled_dataset.csv")
    print(f"\nSaving merged dataset to {output_file}")
    print(f"Total samples: {len(merged_data)}")
    merged_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    merge_data()
