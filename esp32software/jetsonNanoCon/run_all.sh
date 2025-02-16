#!/bin/bash
# run_all.sh
# Usage: ./run_all.sh [x] [y] [state] [version]

set -euo pipefail

# Check arguments
if [ $# -ne 4 ]; then
    echo "Usage: ./run_all.sh [x] [y] [state] [version]"
    echo "Example: ./run_all.sh -2 3 0 1    # For absent data at (-2,3)"
    exit 1
fi

# Validate inputs
if [[ "$3" != "0" && "$3" != "1" ]]; then
    echo "State must be 0 (absent) or 1 (present)"
    exit 1
fi

if ! [[ "$4" =~ ^[0-9]+$ ]]; then
    echo "Version must be a positive number"
    exit 1
fi

if ! [[ "$1" =~ ^-?[0-9]+$ ]] || ! [[ "$2" =~ ^-?[0-9]+$ ]]; then
    echo "X and Y coordinates must be numbers (can be negative)"
    exit 1
fi

# Assign arguments to variables
X_COORD="$1"
Y_COORD="$2"
STATE="$3"
VERSION="$4"

# Configuration variables
ESP32_CSI_DIR=~/Desktop/theia/esp32software/jetsonNanoCon
ACTIVE_STA_DIR="$ESP32_CSI_DIR/active_sta"
ACTIVE_AP_DIR="$ESP32_CSI_DIR/active_ap"
TX_PORT="/dev/ttyUSB1"
RX_PORT="/dev/ttyUSB0"
BAUD_RATE="921600"
CAPTURE_INTERVAL=0.5  # Half second interval

# Set up data directories
BASE_DATA_DIR=~/Documents/csi_data/raw
TEMP_DIR=~/Documents/csi_data/temp
mkdir -p "$BASE_DATA_DIR" "$TEMP_DIR"

# Error handling function
function error_exit {
    echo "Error: $1" >&2
    exit 1
}

# Set permissions
sudo chmod 666 /dev/ttyUSB0 2>/dev/null || true
sudo chmod 666 /dev/ttyUSB1 2>/dev/null || true

echo "=== Flashing Transmitter (active_sta) ==="
cd "$ACTIVE_STA_DIR" || error_exit "Cannot change to transmitter directory"
sed -i.bak "s/^CONFIG_ESP_CONSOLE_UART_BAUDRATE=.*/CONFIG_ESP_CONSOLE_UART_BAUDRATE=$BAUD_RATE/" sdkconfig || true
idf.py build flash -p "$TX_PORT" -b "$BAUD_RATE"

echo "=== Flashing Receiver (active_ap) ==="
cd "$ACTIVE_AP_DIR" || error_exit "Cannot change to receiver directory"
sed -i.bak "s/^CONFIG_ESP_CONSOLE_UART_BAUDRATE=.*/CONFIG_ESP_CONSOLE_UART_BAUDRATE=$BAUD_RATE/" sdkconfig || true
idf.py build flash -p "$RX_PORT" -b "$BAUD_RATE"

echo "=== Starting CSI Data Capture ==="
echo "Coordinates: ($X_COORD, $Y_COORD)"
echo "State: $STATE (0=absent, 1=present)"
echo "Version: $VERSION"
echo "Capturing every ${CAPTURE_INTERVAL} seconds"
echo "Press Ctrl+C to stop capture"

# Start capture loop
counter=0
while true; do
    timestamp=$(date +%s%N | cut -b1-13)  # Millisecond timestamp
    OUTPUT_FILE="$BASE_DATA_DIR/csi_${X_COORD}_${Y_COORD}_${STATE}_${VERSION}_${timestamp}.txt"
    TEMP_FILE="$TEMP_DIR/temp_${timestamp}.txt"
    
    # Capture for half a second
    timeout ${CAPTURE_INTERVAL} unbuffer idf.py monitor -p "$RX_PORT" 2>/dev/null | \
    while IFS= read -r line; do
        if [[ "$line" =~ CSI_DATA ]]; then
            echo "$line" | tee -a "$TEMP_FILE"
        fi
    done
    
    # Only keep files that have data
    if [ -f "$TEMP_FILE" ] && [ -s "$TEMP_FILE" ]; then
        mv "$TEMP_FILE" "$OUTPUT_FILE"
        echo "Capture $counter saved to: $OUTPUT_FILE"
        ((counter++))
    else
        rm -f "$TEMP_FILE"
    fi
    
    sleep 0.1  # Small delay to prevent system overload
done