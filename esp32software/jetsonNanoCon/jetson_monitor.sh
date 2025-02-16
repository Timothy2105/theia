#!/bin/bash
# jetson_monitor.sh
# Usage: ./jetson_monitor.sh [x] [y] [state] [version]

set -euo pipefail

# Configuration
RX_PORT="/dev/ttyUSB0"
BAUD_RATE=921600
BASE_DIR=~/Documents/csi_data
DATA_FILE="$BASE_DIR/csi_live.txt"  # Single file for live data

# Create directory
mkdir -p "$BASE_DIR"

# Function to configure serial port
setup_serial() {
    stty -F "$RX_PORT" \
        "$BAUD_RATE" \
        raw \
        -echo \
        cs8 \
        -cstopb \
        -crtscts \
        cread \
        clocal
}

# Main capture function
capture_data() {
    local x=$1
    local y=$2
    local state=$3
    local version=$4
    
    echo "Starting live CSI capture at ($x, $y) - State: $state Version: $version"
    echo "Updating $DATA_FILE every 0.5 seconds"
    echo "Press Ctrl+C to stop"
    
    # Clear the file initially
    > "$DATA_FILE"
    
    while true; do
        # Temporary file for current capture
        temp_data=$(timeout 0.5 dd bs=128 if="$RX_PORT" 2>/dev/null | grep "CSI_DATA")
        
        if [ ! -z "$temp_data" ]; then
            # Update the live file with new data
            echo "$temp_data" > "$DATA_FILE"
            echo "$(date +%H:%M:%S) - Data updated"
        fi
        
        sleep 0.1  # Small delay to prevent CPU overload
    done
}

# Validate inputs
if [ $# -ne 4 ]; then
    echo "Usage: $0 [x] [y] [state] [version]"
    exit 1
fi

# Set permissions
sudo chmod 666 "$RX_PORT" 2>/dev/null || true

# Setup serial port
setup_serial

# Start capture with error handling
while true; do
    if ! capture_data "$1" "$2" "$3" "$4"; then
        echo "Capture failed, retrying in 1 second..."
        sleep 1
    fi
done
