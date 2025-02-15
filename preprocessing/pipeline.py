import os
from label import label_data
from merge import merge_data
from clean import clean_csi_data

def run_pipeline():
    """
    Run the complete data processing pipeline:
    1. Label raw CSI data
    2. Merge labelled data
    3. Clean and preprocess merged data
    """
    print("\n=== Starting Data Processing Pipeline ===")
    
    print("\n1. Labelling Raw Data...")
    label_data()
    
    print("\n2. Merging Labelled Data...")
    merge_data()
    
    print("\n3. Cleaning and Preprocessing...")
    clean_csi_data()
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    run_pipeline()
