#!/usr/bin/env python3
"""
Q3 Analysis: Demographic Factors

This script performs analysis for Research Question 3:
"How do demographic factors influence individual variability in electric field optimization?"

It analyzes correlations between demographic factors (age, bone characteristics) 
and electric field measures in regions of interest (ROI).

Author: TI-Toolbox Research Team
Date: July 2024
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add analysis directory to path
sys.path.append(str(Path(__file__).parent))

from data_loader import load_q3_data
from statistical_analysis import analyze_q3_demographics
from plotting_utils import create_q3_plots


def run_q3_analysis(target: str, optimization_type: str, 
                   save_results: bool = True, save_plots: bool = True) -> dict:
    """
    Run complete Q3 analysis for a specific target and optimization type.
    
    Args:
        target (str): Target region ('Left_Insula', 'Right_Hippocampus', 'sphere_x-36.1_y14.14_z0.33')
        optimization_type (str): Optimization type ('max' or 'normal')
        save_results (bool): Whether to save results to file
        save_plots (bool): Whether to save plots to file
        
    Returns:
        dict: Complete analysis results
    """
    print(f"\n{'='*60}")
    print(f"Q3 ANALYSIS: Demographic Factors")
    print(f"Target: {target}")
    print(f"Optimization: {optimization_type}")
    print(f"{'='*60}")
    
    # Step 1: Load data
    print("\n1. Loading data...")
    try:
        df, quality_report = load_q3_data(target, optimization_type)
        print(f"   ✓ Data loaded successfully: {len(df)} participants")
        print(f"   ✓ Data quality: {quality_report['n_participants']} participants, "
              f"{quality_report['missing_values']} missing values")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return None
    
    # Step 2: Perform statistical analysis
    print("\n2. Performing statistical analysis...")
    try:
        analysis_results = analyze_q3_demographics(df, target, optimization_type)
        print("   ✓ Statistical analysis completed")
        
        # Print summary report
        print("\n" + analysis_results['summary_report'])
        
    except Exception as e:
        print(f"   ✗ Error in statistical analysis: {e}")
        return None
    
    # Step 3: Create visualizations
    print("\n3. Creating visualizations...")
    try:
        figures = create_q3_plots(df, target, optimization_type)
        print(f"   ✓ Created {len(figures)} figure(s)")
        
        if save_plots:
            # Create results directory if it doesn't exist
            results_dir = Path(__file__).parent.parent / 'results' / 'figures'
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save figures
            for i, fig in enumerate(figures):
                filename = f"q3_{target}_{optimization_type}_figure_{i+1}.png"
                filepath = results_dir / filename
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"   ✓ Saved: {filename}")
        
    except Exception as e:
        print(f"   ✗ Error creating visualizations: {e}")
    
    # Step 4: Save results
    if save_results:
        print("\n4. Saving results...")
        try:
            # Create results directory if it doesn't exist
            results_dir = Path(__file__).parent.parent / 'results' / 'tables'
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare results for saving
            save_data = {
                'analysis_info': {
                    'analysis_type': 'Q3_demographics',
                    'target': target,
                    'optimization_type': optimization_type,
                    'timestamp': datetime.now().isoformat(),
                    'n_participants': quality_report['n_participants']
                },
                'data_quality': quality_report,
                'correlation_results': analysis_results.get('correlation_results', {}),
                'regression_results': analysis_results.get('regression_results', {}),
                'summary_report': analysis_results['summary_report']
            }
            
            # Save as JSON
            json_filename = f"q3_{target}_{optimization_type}_results.json"
            json_filepath = results_dir / json_filename
            with open(json_filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            print(f"   ✓ Saved: {json_filename}")
            
            # Save summary as text
            txt_filename = f"q3_{target}_{optimization_type}_summary.txt"
            txt_filepath = results_dir / txt_filename
            with open(txt_filepath, 'w') as f:
                f.write(analysis_results['summary_report'])
            print(f"   ✓ Saved: {txt_filename}")
            
        except Exception as e:
            print(f"   ✗ Error saving results: {e}")
    
    print(f"\n{'='*60}")
    print("Q3 ANALYSIS COMPLETED")
    print(f"{'='*60}")
    
    return analysis_results


def run_all_q3_analyses():
    """
    Run Q3 analysis for all available targets and optimization types.
    """
    print("Running Q3 analysis for all available combinations...")
    
    # Define targets and optimization types based on experimental design
    targets = ['Left_Insula', 'Right_Hippocampus']  # Add '10mm_sphere' when available
    optimization_types = ['max', 'normal']
    
    all_results = {}
    
    for target in targets:
        all_results[target] = {}
        
        for opt_type in optimization_types:
            print(f"\nProcessing: {target} - {opt_type}")
            
            try:
                results = run_q3_analysis(target, opt_type)
                all_results[target][opt_type] = results
            except Exception as e:
                print(f"Error processing {target} - {opt_type}: {e}")
                all_results[target][opt_type] = None
    
    return all_results


def main():
    """
    Main function to run Q3 analysis.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Q3 Demographic Factors Analysis')
    parser.add_argument('--target', type=str, 
                                               choices=['Left_Insula', 'Right_Hippocampus', 'sphere_x-36.1_y14.14_z0.33'],
                       help='Target region to analyze')
    parser.add_argument('--optimization', type=str, 
                       choices=['max', 'normal'],
                       help='Optimization type to analyze')
    parser.add_argument('--all', action='store_true',
                       help='Run analysis for all available combinations')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results and plots')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_q3_analyses()
    elif args.target and args.optimization:
        run_q3_analysis(args.target, args.optimization, 
                       save_results=not args.no_save, 
                       save_plots=not args.no_save)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 