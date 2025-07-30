# TI-Toolbox Research Environment Setup

This document provides instructions for setting up the Python virtual environment and installing dependencies for the TI-Toolbox research project.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git (for version control)

## Quick Setup

### Option 1: Automated Setup (Recommended)

#### On macOS/Linux:
```bash
# Make the setup script executable (if not already done)
chmod +x setup_environment.sh

# Run the setup script
./setup_environment.sh
```

#### On Windows:
```cmd
# Run the setup script
setup_environment.bat
```

### Option 2: Manual Setup

#### 1. Create Virtual Environment
```bash
# Create a new virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate.bat
```

#### 2. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

#### 3. Verify Installation
```bash
# Test that all packages are installed correctly
python -c "import pandas, numpy, scipy, statsmodels, matplotlib, seaborn; print('All packages installed successfully!')"
```

## Virtual Environment Management

### Activating the Environment
```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate.bat
```

### Deactivating the Environment
```bash
deactivate
```

### Checking Active Environment
```bash
# Check which Python you're using
which python

# Check installed packages
pip list
```

## Project Structure After Setup

```
TIT_Paper/
├── venv/                      # Virtual environment (created)
├── requirements.txt           # Python dependencies
├── setup_environment.sh       # Setup script for macOS/Linux
├── setup_environment.bat      # Setup script for Windows
├── ENVIRONMENT_SETUP.md       # This file
├── analysis/                  # Research question analyses
├── scripts/                   # Reusable modules
├── data/                      # Data files
└── results/                   # Output files
```

## Running Analyses

Once the environment is set up, you can run the statistical analyses:

```bash
# Activate the virtual environment first
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate.bat  # Windows

# Run all analyses
python scripts/analysis/run_all_analyses.py --all

# Run specific analysis
python scripts/analysis/run_all_analyses.py --target Left_insula --optimization max

# Get help
python scripts/analysis/run_all_analyses.py --help
```

## Dependencies

### Core Dependencies
- **pandas** (≥1.5.0): Data manipulation and analysis
- **numpy** (≥1.21.0): Numerical computing
- **scipy** (≥1.9.0): Scientific computing
- **statsmodels** (≥0.13.0): Statistical analysis
- **matplotlib** (≥3.5.0): Plotting and visualization
- **seaborn** (≥0.11.0): Statistical data visualization

### Optional Dependencies
The following packages are commented out in `requirements.txt` but can be uncommented if needed:

- **simnibs** (≥3.2.0): For TI simulation (requires special installation)
- **plotly** (≥5.0.0): Interactive plotting
- **bokeh** (≥2.4.0): Interactive visualization
- **pingouin** (≥0.5.0): Advanced statistics
- **scikit-learn** (≥1.1.0): Machine learning

## Troubleshooting

### Common Issues

#### 1. Python Not Found
```bash
# Check if Python is installed
python3 --version

# If not installed, install Python 3 from python.org
```

#### 2. Permission Errors
```bash
# On macOS/Linux, you might need to use sudo
sudo python3 -m venv venv
```

#### 3. Package Installation Failures
```bash
# Try upgrading pip first
pip install --upgrade pip

# Install packages one by one to identify the problematic one
pip install pandas
pip install numpy
# etc.
```

#### 4. Virtual Environment Not Activating
```bash
# Check if the activation script exists
ls venv/bin/activate  # macOS/Linux
dir venv\Scripts\activate.bat  # Windows

# Try creating a new virtual environment
rm -rf venv
python3 -m venv venv
```

#### 5. Import Errors
```bash
# Make sure the virtual environment is activated
which python  # Should point to venv/bin/python

# Reinstall packages
pip install --force-reinstall -r requirements.txt
```

### Platform-Specific Issues

#### macOS
- If you get SSL errors, you might need to install certificates:
  ```bash
  pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
  ```

#### Windows
- If you get path length errors, try:
  ```cmd
  pip install --no-cache-dir -r requirements.txt
  ```

#### Linux
- You might need to install development headers:
  ```bash
  sudo apt-get install python3-dev  # Ubuntu/Debian
  sudo yum install python3-devel    # CentOS/RHEL
  ```

## Updating Dependencies

To update packages to their latest versions:

```bash
# Activate virtual environment
source venv/bin/activate

# Update all packages
pip install --upgrade -r requirements.txt

# Or update specific packages
pip install --upgrade pandas numpy scipy
```

## Adding New Dependencies

To add new packages to the project:

1. Install the package in your virtual environment:
   ```bash
   pip install new_package_name
   ```

2. Add it to `requirements.txt`:
   ```
   new_package_name>=1.0.0
   ```

3. Commit the changes:
   ```bash
   git add requirements.txt
   git commit -m "Add new_package_name dependency"
   ```

## Environment Variables (Optional)

You can set environment variables for the project:

```bash
# Create a .env file in the root directory
echo "PYTHONPATH=./scripts" > .env
echo "MATPLOTLIB_BACKEND=Agg" >> .env
```

## Best Practices

1. **Always activate the virtual environment** before running analyses
2. **Keep the virtual environment in version control** (venv folder should be in .gitignore)
3. **Update requirements.txt** when adding new dependencies
4. **Use specific version numbers** in requirements.txt for reproducibility
5. **Test the environment** on different machines to ensure portability

## Support

If you encounter issues not covered in this document:

1. Check the [Python venv documentation](https://docs.python.org/3/library/venv.html)
2. Review the [pip documentation](https://pip.pypa.io/en/stable/)
3. Check package-specific documentation for installation issues
4. Create an issue in the project repository

---

**Last Updated**: July 2024  
**Version**: 1.0 