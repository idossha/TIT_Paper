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
│   │   ├── run_q1_analysis.py # Q1: Individualization effects
│   │   ├── run_q2_analysis.py # Q2: Mapping effects
│   │   ├── run_q3_analysis.py # Q3: Demographic factors
│   │   └── run_all_analyses.py # Master script for all analyses
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

1. **Setup environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run all analyses**:
   ```bash
   cd case_studies
   python analysis/run_all_analyses.py --target Left_insula --optimization max
   ```

3. **Run specific research questions**:
   ```bash
   # Q1: Individualization effects
   python analysis/run_all_analyses.py --target Left_Insula --optimization max --questions Q1
   
   # Q2: Mapping effects
   python analysis/run_all_analyses.py --target Left_Insula --optimization max --questions Q2
   
   # Q3: Demographic factors
   python analysis/run_all_analyses.py --target Left_Insula --optimization max --questions Q3
   ```

4. **Run comprehensive pairwise comparisons**:
   ```bash
   # All three comparisons for a target
   python analysis/run_all_comparisons.py --target Left_Insula --optimization max
   
   # Specific comparison
   python analysis/run_all_comparisons.py --target Left_Insula --optimization max --condition-a ernie --condition-b mapped
   
   # All comparisons for all targets
   python analysis/run_all_comparisons.py --all-targets
   ```

5. **Run individual analysis scripts**:
   ```bash
   python analysis/run_q3_analysis.py --target Left_Insula --optimization max
   python analysis/run_all_comparisons.py --target Left_Insula --optimization max
   ```

## 📊 Data Files

### Processed Data (`case_studies/data/processed/`)
- `demographics.csv` - Participant demographic information

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

### Usage Examples
```bash
# ROI targets
python analysis/run_all_analyses.py --target Left_Insula --optimization max
python analysis/run_all_analyses.py --target Right_Hippocampus --optimization normal

# Spherical targets
python analysis/run_all_analyses.py --target sphere_x-36.1_y14.14_z0.33 --optimization max
python analysis/run_all_comparisons.py --target sphere_x-36.1_y14.14_z0.33 --optimization max
```

## 🔧 Key Scripts

### Analysis Scripts
- `case_studies/analysis/run_all_analyses.py` - Master script for running Q3 and comprehensive comparisons
- `case_studies/analysis/run_all_comparisons.py` - Comprehensive pairwise comparisons (replaces Q1/Q2)
- `case_studies/analysis/run_q3_analysis.py` - Q3: Demographic factors analysis

### Supporting Modules
- `case_studies/analysis/data_loader.py` - Data loading utilities
- `case_studies/analysis/statistical_analysis.py` - Statistical analysis functions
- `case_studies/analysis/plotting_utils.py` - Visualization utilities
- `toolbox_related/scripts/TI_simple_test.py` - Toolbox testing utilities
- `requirements.txt` - Python dependencies

## 📈 Results & Visualizations

### Figures (`case_studies/results/figures/`)
- Generated plots from Q3 analysis and comprehensive comparisons
- Publication-ready visualizations with statistical information

### Tables (`case_studies/results/tables/`)
- Statistical results in JSON format
- Summary reports in text format
- Complete analysis results

### Toolbox Resources (`toolbox_related/`)
- `scripts/` - Toolbox-related utilities
- `imgs/` - Toolbox diagrams and images
- `ernie_data/` - ERNIE atlas data and annotations

## 🎯 Research Workflow

1. **Data Preparation**: Raw data → `case_studies/data/processed/`
2. **Q3 Analysis**: Run demographic factors analysis using `case_studies/analysis/` scripts
3. **Comprehensive Comparisons**: Run all pairwise comparisons (ernie vs mapped, ernie vs optimized, mapped vs optimized)
4. **Results**: Generated figures and tables in `case_studies/results/`
5. **Toolbox Development**: Toolbox methods and utilities in `toolbox_related/`

## 🤝 Contributing

For questions about the research methodology or analysis approach, please refer to the research questions section above. The experimental design table provides a clear overview of all analyses performed.

---

**Last Updated**: July 2024
**Paper Status**: In Preparation
