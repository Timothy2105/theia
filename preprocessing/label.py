import pandas as pd
import os
import re

def label_data():
	"""Process and label raw CSI data files"""
	# Create labelled-data directory if it doesn't exist
	os.makedirs('labelled-data', exist_ok=True)
	
	# Get all CSV files from raw-data directory
	raw_data_dir = 'raw-data'
	csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
	
	print(f"\nFound {len(csv_files)} CSV files:")
	for i, file in enumerate(csv_files, 1):
		print(f"{i}. {file}")

	for i, csv_in in enumerate(csv_files, 1):
		# Extract x, y, state from filename (raw-x-y-state-version.csv)
		match = re.match(r'raw-([n\d]+)-([n\d]+)-(\d+)-(\d+)\.csv', csv_in)
		if match:
			x_str, y_str, state, version = match.groups()
			x = -1 if x_str == 'n' else int(x_str)
			y = -1 if y_str == 'n' else int(y_str)
			state = int(state)
			version = int(version)
			
			print(f"\nFile {i}/{len(csv_files)}: {csv_in}")
			print(f"Extracted: x={x}, y={y}, state={state}, version={version}")
		else:
			print(f"\nSkipping {csv_in} - doesn't match expected format")
			continue

		input_path = os.path.join(raw_data_dir, csv_in)
		output_path = os.path.join('labelled-data', f"labelled-{x_str}-{y_str}-{state}-{version}.csv")
		
		print(f"Processing...")
		data = pd.read_csv(input_path)

		data["locationX"] = x
		data["locationY"] = y
		data["state"] = state

		want = data[["CSI_DATA", "state", "locationX", "locationY"]]
		print(want.head())

		print(f"Writing to {output_path}")
		want.to_csv(output_path, index=False)

if __name__ == "__main__":
	label_data()
