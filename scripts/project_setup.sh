#!/bin/bash

# setup.sh
# Scaffolding for Multi-Modal Document Classification project

echo "🚀 Starting project setup..."

# 1. Define folder structure
# We include .gitkeep in folders we want to track in Git without tracking their contents
folders=(
    "data/raw"
    "data/processed"
    "models/pretrained"
    "models/checkpoints"
    "mlruns"
    "logs"
    "dev_scripts"
)

echo "Creating directory tree..."
for dir in "${folders[@]}"; do
    mkdir -p "$dir"
    echo "Created: $dir"
done


# 2. Check for 'uv' and Sync
echo "Checking for 'uv'..."
if command -v uv &> /dev/null; then
    echo "'uv' detected. Synchronizing environment..."
    uv sync
else
    echo "'uv' not found in PATH."
    echo "Install it with: pip install uv ..."
    echo "Oce installed, run `uv sync` to install required dependencies."
fi

# 3. Copy .env.template file as .env file
cp .env.template .env


echo "Setup complete. Your data/ and models/ folders are ready (and ignored by git)."