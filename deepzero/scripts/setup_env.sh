#!/bin/bash

# Create a new Conda environment for the DeepZero project
conda create -n deepzero python=3.12 -y

# Activate the newly created environment
conda activate deepzero

# Upgrade pip to the latest version
pip install --upgrade pip

# Install the required dependencies from requirements.txt
pip install -r requirements.txt

# Instructions for checking CUDA Toolkit installation
echo "Please ensure that the appropriate CUDA Toolkit is installed (e.g., 11.x)."
echo "You can verify the installation by running: nvcc --version"

# Instructions for installing CuPy
echo "To install CuPy for CUDA 11.x, run the following command:"
echo "pip install cupy-cuda11x"