#!/bin/bash

# List of logs
LOGS=(
    "log/Town10HD/recording_20260410_152712_best"
    "log/Town01/recording_20260308_212005"
    "log/Town01/recording_20260204_010805"
    "log/Town02/recording_20260329_233141_best" 
    "log/Town03/recording_20260323_200940_best"
    "log/Town03/recording_20260317_214033_best"
    "log/Town04/recording_20260329_164940_best"
    "log/Town04/recording_20260410_154404_best"
    "log/Town05/recording_20260318_083409_best"
    "log/Town05/recording_20260323_204100_best"
    "log/Town06/recording_20260323_210357_best"
    "log/Town06/recording_20260410_160255_best"
    "log/Town07/recording_20260317_233603"
)

# Loop through and run each command
for LOG in "${LOGS[@]}"; do
    echo "----------------------------------------------------------------"
    echo "STARTING REPLAY: $LOG"
    echo "----------------------------------------------------------------"
    
    uv run main.py --sync --delay 0.025 --fps 70 --debug --timeout 30.0 \
    replay --replay-dir "$LOG" \
    --collect-data model/Autonomous_Dataset/carla/LAWM2_V2 --headless    
    echo "FINISHED: $LOG"
done
