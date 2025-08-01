#!/usr/bin/env python3
"""
TI-Toolbox Analysis Entry Point

This is the main entry point for running TI-Toolbox analyses.
It provides a simple interface to run analyses using configuration files.

Usage:
    python main.py                    # Run with default settings
    python main.py --help            # Show help
    python main.py --all-regions     # Run all regions
    python main.py --region Left_Insula --optimization max  # Run specific analysis

Author: TI-Toolbox Research Team
Date: July 2024
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import and run the analysis runner
from pipeline import main

if __name__ == "__main__":
    sys.exit(main()) 