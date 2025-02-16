#!/bin/bash

WEIGHTS_PATH="weights/weights_v15.h5"
CSI_DATA_PATH="/home/river/Documents/csi_data/csi_live.txt"
PREDICTION_PATH="/home/river/Documents/csi_data/predictions.txt"

mkdir -p $(dirname "$CSI_DATA_PATH")
mkdir -p $(dirname "$PREDICTION_PATH")

cleanup() {
    echo "Stopping all processes..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

trap cleanup EXIT

echo "Starting CSI monitoring..."
~/Documents/GitHub/theia/esp32software/jetsonNanoCon/jetson_monitor.sh 0 0 1 1 &

sleep 2

echo "Starting model monitoring..."
cd ~/Documents/GitHub/theia/live-demo
python3 monitor-csi.py \
    --input "$CSI_DATA_PATH" \
    --output "$PREDICTION_PATH" \
    --weights "$WEIGHTS_PATH"

# Keep running until Ctrl+C
wait
