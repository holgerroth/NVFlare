#!/bin/bash

# Lumos5G Training Quick Start Script

echo "=== Lumos5G Throughput Prediction Training ==="
echo ""

# Run training
echo ""
echo "Starting training..."
echo ""
python train.py \
    --data_path ./train.csv \
    --output_dir outputs \
    --epochs 50 \
    --batch_size 256 \
    --lr 0.001

echo ""
echo "Training complete! Check the outputs directory for results."

