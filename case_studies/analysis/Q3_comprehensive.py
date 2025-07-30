#!/usr/bin/env python3
"""
Comprehensive Q3 Analysis: Demographic Correlations with All Mapping Types

This script performs correlation analysis between demographic variables and 
electric field values across all three mapping approaches:
- Ernie (generalized)
- Mapped (anatomical mapping)
- Optimized (individual optimization)

Author: TI-Toolbox Research Team
Date: July 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.api import OLS, add_constant
import warnings
warnings.filterwarnings('ignore')

# Import the updated plotting utilities
from plotting_utils import PublicationPlotter, PROFESSIONAL_COLORS

def load_and_prepare_data():
    """Load demographics and field data for all mapping types."""
    
    # Load demographics
    demographics = pd.read_csv('data/processed/demographics.csv')
    
    # Define regions and mapping types
    regions = ['Left_Insula_normal', 'Left_Insula_max', 'Right_Hippocampus_max']
    mapping_types = ['ernie', 'mapped', 'opt']
    
    # Load field data for each region and mapping type
    field_data = {}
    
    for region in regions:
        field_data[region] = {}
        for mapping_type in mapping_types:
            filename = f'data/processed/{region}_{mapping_type}.csv'
            try:
                df = pd.read_csv(filename)
                # Remove AVERAGE row if present
                df = df[df['Subject_ID'] != 'AVERAGE'].copy()
                field_data[region][mapping_type] = df
                print(f"Loaded: {filename}")
            except FileNotFoundError:
                print(f"Warning: {filename} not found")
                field_data[region][mapping_type] = None
    
    return demographics, field_data

def create_comprehensive_correlation_analysis(demographics, field_data, region, save_path=None):
    """
    Create comprehensive correlation analysis for a specific region.
    
    Args:
        demographics (pd.DataFrame): Demographics data
        field_data (dict): Field data for all mapping types
        region (str): Region to analyze
        save_path (str): Path to save figures
    """
    
    # Get field data for this region
    region_data = field_data[region]
    
    # Prepare data for analysis
    analysis_data = {}
    mapping_types = ['ernie', 'mapped', 'opt']
    
    for mapping_type in mapping_types:
        if region_data[mapping_type] is not None:
            # Merge demographics with field data
            merged = demographics.merge(region_data[mapping_type], on='Subject_ID', how='inner')
            
            # Select relevant columns (check which ones exist)
            available_cols = merged.columns.tolist()
            field_vars = ['ROI_Mean', 'ROI_Max', 'ROI_Focality']
            demo_vars = ['age', 'bone_volume', 'bone_mean_thick']
            
            # Add Normal columns if they exist
            if 'Normal_Mean' in available_cols:
                field_vars.append('Normal_Mean')
            if 'Normal_Max' in available_cols:
                field_vars.append('Normal_Max')
            
            # Create analysis dataset
            analysis_cols = ['Subject_ID'] + demo_vars + field_vars
            analysis_data[mapping_type] = merged[analysis_cols].copy()
            
            print(f"{region} - {mapping_type}: {len(analysis_data[mapping_type])} subjects")
        else:
            analysis_data[mapping_type] = None
            print(f"{region} - {mapping_type}: No data available")
    
    # Create comprehensive correlation plot
    create_correlation_comparison_plot(analysis_data, region, save_path)
    
    # Create regression comparison plot
    create_regression_comparison_plot(analysis_data, region, save_path)
    
    return analysis_data

def create_correlation_comparison_plot(analysis_data, region, save_path=None):
    """Create correlation comparison plot across all mapping types."""
    
    plotter = PublicationPlotter()
    plotter.setup_publication_style()
    
    # Define variables
    field_vars = ['ROI_Mean', 'ROI_Max', 'ROI_Focality']
    demo_vars = ['age', 'bone_volume', 'bone_mean_thick']
    mapping_types = ['ernie', 'mapped', 'opt']
    
    # Calculate figure size
    n_field_vars = len(field_vars)
    n_demo_vars = len(demo_vars)
    n_mapping_types = len([m for m in mapping_types if analysis_data[m] is not None])
    
    fig_width = max(4 * n_demo_vars, 12)
    fig_height = max(4 * n_field_vars, 10)
    
    fig, axes = plt.subplots(n_field_vars, n_demo_vars, 
                            figsize=(fig_width, fig_height), dpi=300)
    
    if n_field_vars == 1 and n_demo_vars == 1:
        axes = np.array([[axes]])
    elif n_field_vars == 1:
        axes = axes.reshape(1, -1)
    elif n_demo_vars == 1:
        axes = axes.reshape(-1, 1)
    
    # Colors for different mapping types
    mapping_colors = {
        'ernie': PROFESSIONAL_COLORS['secondary'],
        'mapped': PROFESSIONAL_COLORS['accent'],
        'opt': PROFESSIONAL_COLORS['tertiary']
    }
    
    for i, field_var in enumerate(field_vars):
        for j, demo_var in enumerate(demo_vars):
            ax = axes[i, j]
            
            # Plot correlations for each mapping type
            for mapping_type in mapping_types:
                if analysis_data[mapping_type] is not None:
                    data = analysis_data[mapping_type]
                    valid_data = data[[field_var, demo_var]].dropna()
                    
                    if len(valid_data) >= 3:
                        # Calculate correlation
                        correlation, p_value = pearsonr(valid_data[demo_var], valid_data[field_var])
                        
                        # Create legend label with r and p-value
                        sig_text = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                        legend_label = f'{mapping_type.upper()} (r={correlation:.3f}{sig_text}, p={p_value:.3f})'
                        
                        # Create scatter plot
                        ax.scatter(valid_data[demo_var], valid_data[field_var], 
                                 alpha=0.7, s=40, color=mapping_colors[mapping_type],
                                 edgecolors='white', linewidth=0.5, zorder=2,
                                 label=legend_label)
                        
                        # Add trend line
                        z = np.polyfit(valid_data[demo_var], valid_data[field_var], 1)
                        p = np.poly1d(z)
                        ax.plot(valid_data[demo_var], p(valid_data[demo_var]), 
                               color=mapping_colors[mapping_type], alpha=0.8, 
                               linewidth=1.5, linestyle='--', zorder=1)
            
            # Customize plot
            ax.set_xlabel(demo_var.replace('_', ' ').title(), 
                        fontsize=12, fontweight='bold', color=PROFESSIONAL_COLORS['primary'])
            ax.set_ylabel(field_var.replace('_', ' ').title(), 
                        fontsize=12, fontweight='bold', color=PROFESSIONAL_COLORS['primary'])
            
            # Add grid and styling
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            
            ax.tick_params(axis='both', labelsize=10, color=PROFESSIONAL_COLORS['primary'])
            
            # Add legend to all plots
            ax.legend(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.02))
    
    # Add overall title
    fig.suptitle(f'Q3: Demographic Correlations - {region.replace("_", " ").title()}', 
                fontsize=16, fontweight='bold', color=PROFESSIONAL_COLORS['primary'], y=1.08)
    
    # Adjust layout to make room for legends
    plt.tight_layout()
    plt.subplots_adjust(top=0.75, hspace=0.4, wspace=0.3)
    
    if save_path:
        fig.savefig(f"{save_path}/q3_correlations_{region}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

def create_regression_comparison_plot(analysis_data, region, save_path=None):
    """Create regression comparison plot across all mapping types."""
    
    plotter = PublicationPlotter()
    plotter.setup_publication_style()
    
    # Define variables
    target_var = 'ROI_Mean'
    predictor_vars = ['age', 'bone_volume', 'bone_mean_thick']
    mapping_types = ['ernie', 'mapped', 'opt']
    
    # Colors for different mapping types
    mapping_colors = {
        'ernie': PROFESSIONAL_COLORS['secondary'],
        'mapped': PROFESSIONAL_COLORS['accent'],
        'opt': PROFESSIONAL_COLORS['tertiary']
    }
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # Panel A: R² comparison
    r_squared_values = []
    mapping_labels = []
    
    for mapping_type in mapping_types:
        if analysis_data[mapping_type] is not None:
            data = analysis_data[mapping_type]
            valid_data = data[[target_var] + predictor_vars].dropna()
            
            if len(valid_data) >= len(predictor_vars) + 2:
                # Fit regression model
                X = add_constant(valid_data[predictor_vars])
                y = valid_data[target_var]
                model = OLS(y, X).fit()
                
                r_squared_values.append(model.rsquared)
                mapping_labels.append(mapping_type.upper())
    
    # Plot R² comparison
    bars = ax1.bar(mapping_labels, r_squared_values, 
                   color=[mapping_colors[m.lower()] for m in mapping_labels],
                   alpha=0.8, edgecolor=PROFESSIONAL_COLORS['primary'], linewidth=1.2)
    
    # Add R² values on bars
    for bar, r2 in zip(bars, r_squared_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{r2:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('R²', fontsize=12, fontweight='bold', color=PROFESSIONAL_COLORS['primary'])
    ax1.set_title('A. Model Fit Comparison', fontsize=13, fontweight='bold',
                  color=PROFESSIONAL_COLORS['primary'])
    ax1.tick_params(axis='both', labelsize=10, color=PROFESSIONAL_COLORS['primary'])
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_linewidth(1.2)
    
    # Panel B: Coefficient comparison
    all_coefficients = []
    all_predictors = []
    all_mapping_types = []
    
    for mapping_type in mapping_types:
        if analysis_data[mapping_type] is not None:
            data = analysis_data[mapping_type]
            valid_data = data[[target_var] + predictor_vars].dropna()
            
            if len(valid_data) >= len(predictor_vars) + 2:
                # Fit regression model
                X = add_constant(valid_data[predictor_vars])
                y = valid_data[target_var]
                model = OLS(y, X).fit()
                
                # Extract coefficients (excluding intercept)
                coefs = model.params[1:]
                p_vals = model.pvalues[1:]
                
                for pred, coef, p_val in zip(predictor_vars, coefs, p_vals):
                    all_coefficients.append(coef)
                    all_predictors.append(pred.replace('_', ' ').title())
                    all_mapping_types.append(mapping_type.upper())
    
    # Create coefficient comparison plot
    x_pos = np.arange(len(all_coefficients))
    colors = [mapping_colors[m.lower()] for m in all_mapping_types]
    
    bars = ax2.bar(x_pos, all_coefficients, color=colors, alpha=0.8,
                   edgecolor=PROFESSIONAL_COLORS['primary'], linewidth=1.2)
    
    # Add coefficient values
    for bar, coef in zip(bars, all_coefficients):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
                f'{coef:.4f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=8, fontweight='bold')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{pred}\n({m})' for pred, m in zip(all_predictors, all_mapping_types)],
                        rotation=45, fontsize=9, color=PROFESSIONAL_COLORS['primary'])
    ax2.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold', 
                   color=PROFESSIONAL_COLORS['primary'])
    ax2.set_title('B. Regression Coefficients', fontsize=13, fontweight='bold',
                  color=PROFESSIONAL_COLORS['primary'])
    ax2.axhline(y=0, color=PROFESSIONAL_COLORS['primary'], linestyle='-', 
               alpha=0.5, linewidth=1)
    ax2.tick_params(axis='y', labelsize=10, color=PROFESSIONAL_COLORS['primary'])
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=mapping_colors['ernie'], alpha=0.8, label='ERNIE'),
        plt.Rectangle((0,0),1,1, facecolor=mapping_colors['mapped'], alpha=0.8, label='MAPPED'),
        plt.Rectangle((0,0),1,1, facecolor=mapping_colors['opt'], alpha=0.8, label='OPTIMIZED')
    ]
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.05), 
               ncol=3, fontsize=11, frameon=False)
    
    # Add overall title
    fig.suptitle(f'Q3: Regression Analysis - {region.replace("_", " ").title()}', 
                fontsize=16, fontweight='bold', color=PROFESSIONAL_COLORS['primary'], y=1.08)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.85)
    
    if save_path:
        fig.savefig(f"{save_path}/q3_regression_{region}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

def main():
    """Main function to run comprehensive Q3 analysis."""
    
    print("Comprehensive Q3 Analysis: Demographic Correlations with All Mapping Types")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    demographics, field_data = load_and_prepare_data()
    
    # Define regions to analyze
    regions = ['Left_Insula_normal', 'Left_Insula_max', 'Right_Hippocampus_max']
    
    # Create results directory
    import os
    results_dir = 'results/figures'
    os.makedirs(results_dir, exist_ok=True)
    
    # Analyze each region
    for region in regions:
        print(f"\nAnalyzing {region}...")
        analysis_data = create_comprehensive_correlation_analysis(
            demographics, field_data, region, results_dir
        )
        
        # Print summary statistics
        print(f"\nSummary for {region}:")
        for mapping_type in ['ernie', 'mapped', 'opt']:
            if analysis_data[mapping_type] is not None:
                n_subjects = len(analysis_data[mapping_type])
                print(f"  {mapping_type.upper()}: {n_subjects} subjects")
            else:
                print(f"  {mapping_type.upper()}: No data available")
    
    print(f"\nAnalysis complete! Results saved to {results_dir}")

if __name__ == "__main__":
    main() 