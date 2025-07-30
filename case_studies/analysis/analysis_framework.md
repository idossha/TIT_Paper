# TI-Toolbox Research Analysis Framework

## Overview

This document describes the modular analysis framework developed for the TI-Toolbox research project. The framework is designed to efficiently analyze the three main research questions while maintaining code reusability, reproducibility, and clear documentation.

## Architecture

### Directory Structure

```
TIT_Paper/
├── analysis/                    # Research question-specific analyses
│   ├── Q1_individualization/    # Individual vs. generalized model analysis
│   │   └── run_q1_analysis.py   # Main Q1 analysis script
│   ├── Q2_mapping/             # Genetic optimization mapping analysis
│   │   └── run_q2_analysis.py   # Main Q2 analysis script
│   └── Q3_demographics/        # Demographic factors analysis
│       └── run_q3_analysis.py   # Main Q3 analysis script
├── scripts/                    # Reusable modules
│   ├── analysis/               # Core analysis modules
│   │   ├── data_loader.py      # Data loading and validation
│   │   ├── statistical_analysis.py # Statistical analysis functions
│   │   ├── run_all_analyses.py # Master analysis script
│   │   └── requirements.txt    # Python dependencies
│   └── visualization/          # Plotting utilities
│       └── plotting_utils.py   # Publication-ready plotting functions
├── data/                       # Data files
│   └── processed/              # Cleaned and processed data
└── results/                    # Output files
    ├── figures/                # Generated plots
    └── tables/                 # Statistical results
```

## Core Modules

### 1. Data Loading (`scripts/analysis/data_loader.py`)

The `DataLoader` class provides standardized data loading operations:

#### Key Features:
- **Automatic file discovery**: Finds available targets and conditions
- **Data validation**: Checks data quality and completeness
- **Flexible loading**: Supports different target/condition combinations
- **Error handling**: Graceful handling of missing files or data

#### Usage:
```python
from scripts.analysis.data_loader import DataLoader, load_q1_data

# Load data for Q1 analysis
df, quality_report = load_q1_data('Left_insula', 'max')

# Or use the DataLoader class directly
loader = DataLoader()
df = loader.load_target_data('Left_insula', 'max', 'ernie')
```

### 2. Statistical Analysis (`scripts/analysis/statistical_analysis.py`)

The `StatisticalAnalyzer` class provides comprehensive statistical functions:

#### Key Features:
- **Paired comparisons**: For Q1 and Q2 analyses
- **Correlation analysis**: For Q3 demographic factors
- **Multiple regression**: For Q3 predictive modeling
- **Effect size calculations**: Cohen's d, percentage changes
- **Automatic reporting**: Generates human-readable summaries

#### Usage:
```python
from scripts.analysis.statistical_analysis import analyze_q1_individualization

# Run Q1 analysis
results = analyze_q1_individualization(df, 'Left_insula', 'max')
print(results['summary_report'])
```

### 3. Visualization (`scripts/visualization/plotting_utils.py`)

The `PublicationPlotter` class creates publication-ready figures:

#### Key Features:
- **Consistent styling**: Publication-quality formatting
- **Multiple plot types**: Paired comparisons, correlations, regressions
- **Automatic saving**: High-resolution output
- **Customizable**: Flexible color schemes and layouts

#### Usage:
```python
from scripts.visualization.plotting_utils import create_q1_plots

# Create Q1 plots
figures = create_q1_plots(df, 'Left_insula', 'max', save_path='results/figures/')
```

## Research Question Analysis Scripts

### Q1: Individualization Effects

**File**: `analysis/Q1_individualization/run_q1_analysis.py`

**Purpose**: Compares individual head models vs. generalized models

**Analysis Flow**:
1. Load data for 'ernie' vs 'optimized' conditions
2. Perform paired t-tests on mean, max, and focality measures
3. Calculate effect sizes and percentage changes
4. Generate publication-ready plots
5. Save results and summary reports

**Usage**:
```bash
# Run specific analysis
python run_q1_analysis.py --target Left_insula --optimization max

# Run all Q1 analyses
python run_q1_analysis.py --all
```

### Q2: Mapping Effects

**File**: `analysis/Q2_mapping/run_q2_analysis.py`

**Purpose**: Compares free optimization vs. mapped hd-EEG net

**Analysis Flow**:
1. Load data for 'optimized' vs 'mapped' conditions
2. Perform paired t-tests on electric field measures
3. Calculate effect sizes and significance
4. Generate comparison plots
5. Save results and reports

**Usage**:
```bash
# Run specific analysis
python run_q2_analysis.py --target Left_insula --optimization max

# Run all Q2 analyses
python run_q2_analysis.py --all
```

### Q3: Demographic Factors

**File**: `analysis/Q3_demographics/run_q3_analysis.py`

**Purpose**: Examines demographic factors and individual variability

**Analysis Flow**:
1. Load target data and demographic information
2. Perform correlation analyses between variables
3. Run multiple regression to predict electric field
4. Generate correlation and regression plots
5. Create comprehensive summary reports

**Usage**:
```bash
# Run specific analysis
python run_q3_analysis.py --target Right_hippocampus --optimization max

# Run all Q3 analyses
python run_q3_analysis.py --all

# Create comprehensive summary
python run_q3_analysis.py --summary
```

## Master Analysis Script

### Complete Analysis Suite

**File**: `scripts/analysis/run_all_analyses.py`

**Purpose**: Unified interface for running all analyses

**Features**:
- Run individual research questions or all at once
- Support for specific targets and optimization types
- Comprehensive result aggregation
- Automatic summary generation

**Usage**:
```bash
# Run complete analysis for specific target
python run_all_analyses.py --target Left_insula --optimization max

# Run specific research questions
python run_all_analyses.py --target Left_insula --optimization max --questions Q1 Q2

# Run all analyses for all targets
python run_all_analyses.py --all

# Create comprehensive summary
python run_all_analyses.py --summary
```

## Data Requirements

### File Naming Convention

Data files should follow this naming pattern:
```
{target}_{optimization_type}_{condition}.csv
```

Examples:
- `Left_insula_max_ernie.csv`
- `Left_insula_max_mapped.csv`
- `Left_insula_max_optimized.csv`
- `Right_hippocampus_max_ernie.csv`

### Required Columns

Each CSV file should contain:
- `Subject_ID`: Participant identifier
- `mean`: Mean electric field
- `max`: Maximum electric field
- `focality`: Focality measure (optional)

### Demographics File

The `demographics.csv` file should contain:
- `Subject_ID`: Participant identifier
- `age`: Participant age
- `sex`: Participant sex
- `volume`: Bone volume measurements
- `mean`: Mean bone thickness

## Output Structure

### Results Directory

```
results/
├── figures/                    # Generated plots
│   ├── q1_*.png              # Q1 analysis plots
│   ├── q2_*.png              # Q2 analysis plots
│   └── q3_*.png              # Q3 analysis plots
└── tables/                    # Statistical results
    ├── q1_*_results.json     # Q1 analysis results
    ├── q2_*_results.json     # Q2 analysis results
    ├── q3_*_results.json     # Q3 analysis results
    ├── complete_analysis_*.json # Complete analysis results
    └── *_summary.txt         # Summary reports
```

### Result File Format

JSON result files contain:
```json
{
  "analysis_info": {
    "analysis_type": "Q1_individualization",
    "target": "Left_insula",
    "optimization_type": "max",
    "timestamp": "2024-07-29T...",
    "n_participants": 20
  },
  "data_quality": {
    "n_participants": 20,
    "missing_values": {...},
    "numeric_columns": [...]
  },
  "statistical_results": {
    "mean": {
      "diff": 0.123,
      "percent_change": 15.2,
      "p_value": 0.045,
      "cohens_d": 0.67,
      "significant": true
    }
  },
  "summary_report": "STATISTICAL ANALYSIS REPORT..."
}
```

## Installation and Setup

### 1. Install Dependencies

```bash
cd scripts/analysis
pip install -r requirements.txt
```

### 2. Verify Data Structure

Ensure your data files are in the correct location:
```
data/processed/
├── demographics.csv
├── Left_insula_max_ernie.csv
├── Left_insula_max_mapped.csv
├── Left_insula_max_optimized.csv
└── ...
```

### 3. Run Analysis

```bash
# Quick start - run default analysis
python scripts/analysis/run_all_analyses.py

# Run specific analysis
python scripts/analysis/run_all_analyses.py --target Left_insula --optimization max

# Run all analyses
python scripts/analysis/run_all_analyses.py --all
```

## Best Practices

### Code Organization
- Keep analysis scripts modular and focused
- Use consistent naming conventions
- Document all functions and classes
- Handle errors gracefully

### Data Management
- Validate data quality before analysis
- Use relative paths for portability
- Save intermediate results for reproducibility
- Version control your analysis scripts

### Results Management
- Save results in structured formats (JSON, CSV)
- Generate human-readable summaries
- Use consistent file naming
- Organize outputs by analysis type

### Reproducibility
- Set random seeds where applicable
- Document all parameters and settings
- Use version-controlled dependencies
- Include example data and scripts

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure scripts directory is in Python path
2. **Missing Data**: Check file naming and location
3. **Memory Issues**: Process large datasets in chunks
4. **Plotting Errors**: Verify matplotlib backend settings

### Debug Mode

Run with verbose output:
```bash
python -v scripts/analysis/run_all_analyses.py --target Left_insula --optimization max
```

### Data Validation

Use the DataLoader to check your data:
```python
from scripts.analysis.data_loader import DataLoader

loader = DataLoader()
print("Available targets:", loader.get_available_targets())
for target in loader.get_available_targets():
    print(f"{target}: {loader.get_available_conditions(target)}")
```

## Future Enhancements

### Planned Features
- Interactive dashboards for result exploration
- Advanced statistical methods (mixed models, Bayesian analysis)
- Automated report generation (PDF, HTML)
- Integration with external statistical software
- Real-time analysis monitoring

### Extensibility
The framework is designed to be easily extensible:
- Add new research questions by creating new analysis modules
- Extend statistical methods in the StatisticalAnalyzer class
- Add new visualization types to the PublicationPlotter class
- Integrate with external data sources

---

**Last Updated**: July 2024  
**Version**: 1.0  
**Author**: TI-Toolbox Research Team 