#!/bin/bash

# Lumos5G Inference Quick Start Script

echo "=== Lumos5G Throughput Prediction Inference ==="
echo ""

# Check if model checkpoint exists
if [ ! -f "outputs/best_model.pth" ]; then
    echo "Error: Model checkpoint not found at outputs/best_model.pth"
    echo "Please train a model first using train.py"
    exit 1
fi

# Check if data file exists
if [ ! -f "Lumos5G-v1.0/Lumos5G-v1.0.csv" ]; then
    echo "Error: Dataset not found at Lumos5G-v1.0/Lumos5G-v1.0.csv"
    echo "Please ensure the dataset is in the correct location."
    exit 1
fi

# Create output directory
mkdir -p inference_outputs

# Run inference
echo "Starting inference..."
echo ""
python inference.py \
    --checkpoint outputs/best_model.pth \
    --data_path Lumos5G-v1.0/Lumos5G-v1.0.csv \
    --output_dir inference_outputs \
    --batch_size 256 \
    --plot

echo ""
echo "Inference complete! Check the inference_outputs directory for results."

