#!/bin/bash

# Define the dataset directory
DATASET_DIR="/vol/bitbucket/mc620/DeepLearningCW1/dataset"

# Check if the dataset directory exists, create if it does not
if [ ! -d "$DATASET_DIR" ]; then
    mkdir -p "$DATASET_DIR"
fi

# Download the datasets
wget -O "$DATASET_DIR/NaturalImageNetTest.zip" "https://zenodo.org/record/5846979/files/NaturalImageNetTest.zip?download=1"
wget -O "$DATASET_DIR/NaturalImageNetTrain.zip" "https://zenodo.org/record/5846979/files/NaturalImageNetTrain.zip?download=1"

# Unzip the datasets
unzip -o "$DATASET_DIR/NaturalImageNetTest.zip" -d "$DATASET_DIR"
unzip -o "$DATASET_DIR/NaturalImageNetTrain.zip" -d "$DATASET_DIR"

# Optional: Clean up the zip files if you don't need them anymore
rm "$DATASET_DIR/NaturalImageNetTest.zip"
rm "$DATASET_DIR/NaturalImageNetTrain.zip"

echo "Dataset downloaded and extracted to $DATASET_DIR"
