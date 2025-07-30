#!/bin/bash

# TI-Toolbox Research Environment Setup Script
# This script creates a virtual environment and installs all dependencies

echo "=========================================="
echo "TI-Toolbox Research Environment Setup"
echo "=========================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo "Verifying installation..."
python -c "import pandas, numpy, scipy, statsmodels, matplotlib, seaborn; print('All core packages installed successfully!')"

echo "=========================================="
echo "Environment setup completed!"
echo "=========================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate the virtual environment:"
echo "  deactivate"
echo ""
echo "To run the analysis scripts:"
echo "  python scripts/analysis/run_all_analyses.py --help"
echo ""
echo "==========================================" 