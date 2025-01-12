#!/bin/bash

# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version; update for GPU if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric dependencies (replace CPU with GPU version if needed)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
pip install torch-geometric
