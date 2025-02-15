import pandas as pd
import os

def count_array_elements(array_str):
    """Count number of elements in a string representation of an array"""
    # Remove brackets and split by spaces, filtering out empty strings
    numbers = [x for x in array_str.strip('[]').split() if x]
    return len(numbers)

def clean_csi_data():
    """
    Clean merged CSI datasets by removing rows where CSI_DATA length != 128
    """
    # Create cleaned-data directory if it doesn't exist
    os.makedirs('cleaned-data', exist_ok=True)
    
    merged_dir = 'merged-data'
    csv_files = [f for f in os.listdir(merged_dir) if f.endswith('.csv')]
    
    print(f"\nFound {len(csv_files)} merged files to clean:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file}")
    
    for i, csv_file in enumerate(csv_files, 1):
        input_path = os.path.join(merged_dir, csv_file)
        print(f"\nProcessing file {i}/{len(csv_files)}: {csv_file}")
        
        # Read the CSV
        df = pd.read_csv(input_path)
        initial_rows = len(df)
        
        # Filter rows where CSI_DATA length is 128
        df = df[df['CSI_DATA'].apply(count_array_elements) == 128]
        
        # Save statistics
        removed_rows = initial_rows - len(df)
        print(f"Initial rows: {initial_rows}")
        print(f"Rows with incorrect CSI size: {removed_rows}")
        print(f"Remaining rows: {len(df)}")
        
        # Save cleaned data
        output_file = os.path.join('cleaned-data', f"cleaned_{csv_file}")
        print(f"Saving cleaned data to {output_file}")
        df.to_csv(output_file, index=False)

if __name__ == "__main__":
    clean_csi_data()
