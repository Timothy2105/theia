import pandas as pd
import os
import re

def parse_csi_line(line):
	"""Extract CSI data from a line of text"""
	match = re.search(r'\[(.*?)\]', line)
	if match:
		return match.group(0)
	return None

def label_data():
	"""Process and label raw CSI data files from txt format"""
	# Create labelled-data directory if it doesn't exist
	os.makedirs('labelled-data', exist_ok=True)
	
	# Get all TXT files from raw-data directory
	raw_data_dir = 'raw-data'
	txt_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.txt')]
	
	print(f"\nFound {len(txt_files)} TXT files:")
	for i, file in enumerate(txt_files, 1):
		print(f"{i}. {file}")

	for i, txt_in in enumerate(txt_files, 1):
		# Extract x, y, state from filename (raw-x-y-state-version.txt)
		match = re.match(r'csi_(\d+)_(\d+)_(\d+)_(\d+)\.txt', txt_in)
		if match:
			x_str, y_str, state, version = match.groups()
			state = int(state)
			
			# If state is 0 (not present), use 0,0 for coordinates
			# If state is 1 (present), use coordinates from filename
			if state == 0:
				x, y = 0, 0
			else:
				x = int(x_str)
				y = int(y_str)
			
			version = int(version)
			
			print(f"\nFile {i}/{len(txt_files)}: {txt_in}")
			print(f"Extracted: x={x}, y={y}, state={state}, version={version}")
		else:
			print(f"\nSkipping {txt_in} - doesn't match expected format")
			continue

		input_path = os.path.join(raw_data_dir, txt_in)
		output_path = os.path.join('labelled-data', f"labelled_{x_str}_{y_str}_{state}_{version}.csv")
		
		print(f"Processing...")
		
		# Read and process the text file
		csi_data = []
		with open(input_path, 'r') as f:
			for line in f:
				csi_str = parse_csi_line(line)
				if csi_str:
					csi_data.append(csi_str)
		
		# Create DataFrame
		data = pd.DataFrame({
			'CSI_DATA': csi_data,
			'locationX': x,
			'locationY': y,
			'state': state
		})

		print(data.head())
		print(f"Writing to {output_path}")
		data.to_csv(output_path, index=False)

if __name__ == "__main__":
	label_data()
