# TIT_Paper: TI-Toolbox Research Repository

This repository contains scripts, data, and analysis tools for the research paper **"TI-Toolbox: A Comprehensive Software Platform for Temporal Interference Stimulation Research"**.

## 📋 Research Overview

### Primary Research Questions

#### Q1: Does individualization matter for optimizing electrode montage?
- **Main Question**: Does using individual head models vs. generalized models affect electric field optimization in regions of interest (ROI)?
- **Sub-question 1**: Is individualization more important for small cortical targets vs. large cortical/sub-cortical targets?
- **Sub-question 2**: Is individualization more critical when maximizing normal component vs. vector magnitude?

#### Q2: Is mapping genetic optimization output to hd-EEG net acceptable?
- **Main Question**: Are there significant differences between true genetic montage vs. mapped hd-EEG net versions?
- **Sub-question 1**: Does mapping affect certain targets more than others?
- **Sub-question 2**: Is mapping worse when maximizing normal component vs. vector magnitude?

#### Q3: Demographic factors and individual variability
- **Main Question**: Do demographic factors (age, sex, cortical bone mass) explain individual variability?
- **Sub-question**: Do these factors cause higher inter-individual variability in certain ROIs?

## 🎯 Experimental Design

### Targets & Optimization Goals

| Target | Optimization Goal | Q1 Analysis | Q2 Analysis | Q3 Analysis |
|--------|------------------|-------------|-------------|-------------|
| Left Insula | Max Mean Field | Individual vs Generalized | Free vs Mapped | Demographic factors |
| Left Insula | Max Normal | Individual vs Generalized | Free vs Mapped | - |
| 10mm sphere | Max Mean Field | Individual vs Generalized | Free vs Mapped | Demographic factors |
| 10mm sphere | Max Normal | Individual vs Generalized | Free vs Mapped | - |
| R hippocampus | Max Mean Field | Individual vs Generalized | Free vs Mapped | Demographic factors |

**Note**: Max Normal optimization not performed for hippocampus (sub-cortical) as it's not clinically relevant.

## 📁 Repository Structure

```
TIT_Paper/
├── case_studies/              # Case study data, analysis, and results
│   ├── data/                  # Data files
│   │   └── processed/         # Preprocessed data files
│   ├── analysis/              # Analysis scripts and modules
│   │   ├── data_loader.py     # Data loading utilities
│   │   ├── statistical_analysis.py # Statistical analysis functions
│   │   ├── plotting_utils.py  # Visualization utilities
│   │   ├── Q3.py             # Q3 analysis (legacy)
│   │   ├── Q3_comprehensive.py # Q3 comprehensive analysis
│   │   ├── run_q3_analysis.py # Q3: Demographic factors
│   │   ├── run_all_comparisons.py # Comprehensive pairwise comparisons
│   │   ├── run_all_analyses.py # Master script for all analyses
│   │   ├── test_professional_plots.py # Test plotting capabilities
│   │   ├── setup_environment.sh # Environment setup (Unix)
│   │   ├── setup_environment.bat # Environment setup (Windows)
│   │   └── ENVIRONMENT_SETUP.md # Environment setup documentation
│   └── results/               # Analysis outputs
│       ├── figures/           # Generated plots and visualizations
│       └── tables/            # Statistical results and summaries
├── toolbox_related/           # Toolbox methods and utilities
│   ├── scripts/               # Toolbox-related scripts
│   ├── imgs/                  # Toolbox images and diagrams
│   └── ernie_data/            # ERNIE atlas data and annotations
├── drawio/                    # Draw.io diagrams and schematics
├── venv/                      # Python virtual environment
└── requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

**Option A: Using setup scripts (Recommended)**
```bash
# Unix/Linux/macOS
cd case_studies/analysis
chmod +x setup_environment.sh
./setup_environment.sh

# Windows
cd case_studies\analysis
setup_environment.bat
```

**Option B: Manual setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Analyses

**From the project root directory:**

```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run all analyses for a specific target
python case_studies/analysis/run_all_analyses.py --target Left_Insula --optimization max

# Run comprehensive pairwise comparisons
python case_studies/analysis/run_all_comparisons.py --target Left_Insula --optimization max

# Run Q3 demographic analysis
python case_studies/analysis/run_q3_analysis.py --target Left_Insula --optimization max

# Test plotting capabilities
python case_studies/analysis/test_professional_plots.py
```

**From the case_studies directory:**

```bash
cd case_studies

# Run all analyses for a specific target
python analysis/run_all_analyses.py --target Left_Insula --optimization max

# Run comprehensive pairwise comparisons
python analysis/run_all_comparisons.py --target Left_Insula --optimization max

# Run Q3 demographic analysis
python analysis/run_q3_analysis.py --target Left_Insula --optimization max

# Test plotting capabilities
python analysis/test_professional_plots.py
```

### 3. Available Targets and Optimizations

**ROI Targets:**
- `Left_Insula` - Left insula region
- `Right_Hippocampus` - Right hippocampus region

**Spherical Targets:**
- `sphere_x-36.1_y14.14_z0.33` - Spherical target at specific coordinates

**Optimization Types:**
- `max` - Maximize electric field magnitude
- `normal` - Optimize normal component

**Usage Examples:**
```bash
# ROI targets
python case_studies/analysis/run_all_analyses.py --target Left_Insula --optimization max
python case_studies/analysis/run_all_analyses.py --target Right_Hippocampus --optimization normal

# Spherical targets
python case_studies/analysis/run_all_analyses.py --target sphere_x-36.1_y14.14_z0.33 --optimization max

# All comparisons for all targets
python case_studies/analysis/run_all_comparisons.py --all-targets
```

## 📊 Data Files

### Processed Data (`case_studies/data/processed/`)
- `demographics.csv` - Participant demographic information
- `demographics_fake_data.csv` - Synthetic demographic data for testing

### Data Naming Convention

**ROI Targets:** `{hemisphere}_{ROI}_{goal}_{montage}.csv`
- `Left_Insula_max_ernie.csv` - Left insula, maximizing magnitude, ERNIE montage
- `Left_Insula_normal_opt.csv` - Left insula, optimizing normal component, optimized montage
- `Right_Hippocampus_max_mapped.csv` - Right hippocampus, maximizing magnitude, mapped montage

**Spherical Targets:** `sphere_{x}_{y}_{z}_{goal}_{montage}.csv`
- `sphere_x-36.1_y14.14_z0.33_max_ernie.csv` - Spherical target at coordinates (-36.1, 14.14, 0.33), maximizing magnitude, ERNIE montage
- `sphere_x-36.1_y14.14_z0.33_normal_opt.csv` - Spherical target at coordinates (-36.1, 14.14, 0.33), optimizing normal component, optimized montage

**Parameters:**
- **ROI targets**: `Left_Insula`, `Right_Hippocampus`
- **Spherical targets**: `sphere_x-36.1_y14.14_z0.33` (with actual coordinates)
- **goal**: `max` (maximize magnitude), `normal` (optimize normal component)
- **montage**: `ernie` (standardized), `mapped` (hd-EEG mapped), `opt` (optimized)

### Available Datasets
- **Left_Insula**: max/normal optimization with ernie/mapped/opt montages
- **Right_Hippocampus**: max/normal optimization with ernie/mapped/opt montages
- **Spherical targets**: Coordinate-specific targets with max/normal optimization

## 🔧 Key Scripts

### Analysis Scripts
- `case_studies/analysis/run_all_analyses.py` - Master script for running Q3 and comprehensive comparisons
- `case_studies/analysis/run_all_comparisons.py` - Comprehensive pairwise comparisons (replaces Q1/Q2)
- `case_studies/analysis/run_q3_analysis.py` - Q3: Demographic factors analysis
- `case_studies/analysis/Q3_comprehensive.py` - Comprehensive Q3 analysis with all mapping types
- `case_studies/analysis/test_professional_plots.py` - Test plotting capabilities

### Supporting Modules
- `case_studies/analysis/data_loader.py` - Data loading utilities with automatic path resolution
- `case_studies/analysis/statistical_analysis.py` - Statistical analysis functions
- `case_studies/analysis/plotting_utils.py` - Visualization utilities with publication-ready styling
- `toolbox_related/scripts/TI_simple_test.py` - Toolbox testing utilities

### Setup Scripts
- `case_studies/analysis/setup_environment.sh` - Environment setup for Unix/Linux/macOS
- `case_studies/analysis/setup_environment.bat` - Environment setup for Windows
- `case_studies/analysis/ENVIRONMENT_SETUP.md` - Detailed environment setup documentation

## 📈 Results & Visualizations

### Figures (`case_studies/results/figures/`)
- Generated plots from Q3 analysis and comprehensive comparisons
- Publication-ready visualizations with statistical information
- Test plots for verifying plotting capabilities

### Tables (`case_studies/results/tables/`)
- Statistical results in JSON format
- Summary reports in text format
- Complete analysis results

### Toolbox Resources (`toolbox_related/`)
- `scripts/` - Toolbox-related utilities
- `imgs/` - Toolbox diagrams and images
- `ernie_data/` - ERNIE atlas data and annotations

## 🎯 Research Workflow

1. **Environment Setup**: Use setup scripts or manual installation
2. **Data Preparation**: Raw data → `case_studies/data/processed/`
3. **Q3 Analysis**: Run demographic factors analysis using `case_studies/analysis/` scripts
4. **Comprehensive Comparisons**: Run all pairwise comparisons (ernie vs mapped, ernie vs optimized, mapped vs optimized)
5. **Results**: Generated figures and tables in `case_studies/results/`
6. **Toolbox Development**: Toolbox methods and utilities in `toolbox_related/`

## 🔍 Path Handling & I/O

The analysis scripts use robust path handling with the following features:

- **Automatic Path Resolution**: Scripts automatically find data and results directories relative to their location
- **Cross-Platform Compatibility**: Uses `pathlib.Path` for consistent path handling across operating systems
- **Directory Creation**: Automatically creates results directories if they don't exist
- **Error Handling**: Graceful error handling for missing files or directories

### Path Structure
```
case_studies/
├── analysis/          # Scripts location
├── data/processed/    # Input data
└── results/          # Output results
    ├── figures/      # Generated plots
    └── tables/       # Statistical results
```

## 🤝 Contributing

For questions about the research methodology or analysis approach, please refer to the research questions section above. The experimental design table provides a clear overview of all analyses performed.

### Development Guidelines
- All scripts use consistent path handling with `pathlib.Path`
- Results are automatically saved to appropriate subdirectories
- Error handling is implemented for missing data or directories
- Cross-platform compatibility is maintained

---

**Last Updated**: December 2024
**Paper Status**: In Preparation
