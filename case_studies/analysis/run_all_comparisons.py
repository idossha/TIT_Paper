#!/usr/bin/env python3
"""
Comprehensive Comparison Analysis

This script performs all three pairwise comparisons for each target and optimization type:
1. ernie vs mapped
2. ernie vs optimized  
3. mapped vs optimized

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

from data_loader import load_comparison_data
from statistical_analysis import StatisticalAnalyzer
from plotting_utils import PublicationPlotter


def run_comparison_analysis(target: str, optimization_type: str, 
                           condition_a: str, condition_b: str,
                           save_results: bool = True, save_plots: bool = True) -> dict:
    """
    Run comparison analysis between two conditions.
    
    Args:
        target (str): Target region
        optimization_type (str): Optimization type
        condition_a (str): First condition
        condition_b (str): Second condition
        save_results (bool): Whether to save results
        save_plots (bool): Whether to save plots
        
    Returns:
        dict: Analysis results
    """
    comparison_name = f"{condition_a}_vs_{condition_b}"
    
    print(f"\n{'='*60}")
    print(f"COMPARISON: {comparison_name.upper()}")
    print(f"Target: {target}")
    print(f"Optimization: {optimization_type}")
    print(f"{'='*60}")
    
    # Step 1: Load data
    print("\n1. Loading data...")
    try:
        df, quality_report = load_comparison_data(target, optimization_type, condition_a, condition_b)
        print(f"   ✓ Data loaded successfully: {len(df)} participants")
        print(f"   ✓ Data quality: {quality_report['n_participants']} participants, "
              f"{quality_report['missing_values']} missing values")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return None
    
    # Step 2: Perform statistical analysis
    print("\n2. Performing statistical analysis...")
    try:
        analyzer = StatisticalAnalyzer()
        comparison_results = analyzer.paired_comparison_analysis(df)
        
        # Create summary report
        summary_report = create_comparison_summary_report(
            comparison_results, target, optimization_type, condition_a, condition_b
        )
        
        analysis_results = {
            'comparison_results': comparison_results,
            'summary_report': summary_report
        }
        
        print("   ✓ Statistical analysis completed")
        print("\n" + summary_report)
        
    except Exception as e:
        print(f"   ✗ Error in statistical analysis: {e}")
        return None
    
    # Step 3: Create visualizations
    print("\n3. Creating visualizations...")
    try:
        plotter = PublicationPlotter()
        fig = plotter.create_paired_comparison_plot(
            df, title=f"Comparison: {condition_a} vs {condition_b} - {target} ({optimization_type})",
            stats_results=comparison_results
        )
        
        print(f"   ✓ Created 1 figure")
        
        if save_plots:
            # Create results directory if it doesn't exist
            results_dir = Path(__file__).parent.parent / 'results' / 'figures'
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save figure
            filename = f"comparison_{comparison_name}_{target}_{optimization_type}.png"
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
                    'analysis_type': 'comparison',
                    'comparison': comparison_name,
                    'target': target,
                    'optimization_type': optimization_type,
                    'condition_a': condition_a,
                    'condition_b': condition_b,
                    'timestamp': datetime.now().isoformat(),
                    'n_participants': quality_report['n_participants']
                },
                'data_quality': quality_report,
                'statistical_results': comparison_results,
                'summary_report': summary_report
            }
            
            # Save as JSON
            json_filename = f"comparison_{comparison_name}_{target}_{optimization_type}_results.json"
            json_filepath = results_dir / json_filename
            with open(json_filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            print(f"   ✓ Saved: {json_filename}")
            
            # Save summary as text
            txt_filename = f"comparison_{comparison_name}_{target}_{optimization_type}_summary.txt"
            txt_filepath = results_dir / txt_filename
            with open(txt_filepath, 'w') as f:
                f.write(summary_report)
            print(f"   ✓ Saved: {txt_filename}")
            
        except Exception as e:
            print(f"   ✗ Error saving results: {e}")
    
    print(f"\n{'='*60}")
    print(f"COMPARISON {comparison_name.upper()} COMPLETED")
    print(f"{'='*60}")
    
    return analysis_results


def create_comparison_summary_report(results: dict, target: str, optimization_type: str,
                                   condition_a: str, condition_b: str) -> str:
    """
    Create a summary report for comparison analysis.
    
    Args:
        results (dict): Statistical results
        target (str): Target region
        optimization_type (str): Optimization type
        condition_a (str): First condition
        condition_b (str): Second condition
        
    Returns:
        str: Formatted summary report
    """
    report = f"""STATISTICAL ANALYSIS REPORT
==================================================

Analysis Type: Comparison
Target: {target}
Optimization: {optimization_type}
Comparison: {condition_a} vs {condition_b}

Study Design: Within-subjects comparison
Variables Analyzed: {', '.join(results.keys())}

RESULTS:
------------------------------"""

    for var, stats in results.items():
        direction = "higher" if stats['diff'] > 0 else "lower"
        percent_change = stats['percent_change']
        p_val = stats['p_value']
        cohens_d = stats['cohens_d']
        significant = "**SIGNIFICANT**" if stats['significant'] else "Not significant"
        
        report += f"\n{var.upper()}: {direction} in condition B ({stats['diff']:.3f} = {percent_change:+.1f}% = {stats['z_score']:+.2f} SD)"
        report += f"\n  p = {p_val:.3f}, Cohen's d = {cohens_d:.2f} - {significant}\n"

    return report


def run_all_comparisons_for_target(target: str, optimization_type: str,
                                  save_results: bool = True, save_plots: bool = True) -> dict:
    """
    Run all three comparisons for a specific target and optimization type.
    
    Args:
        target (str): Target region
        optimization_type (str): Optimization type
        save_results (bool): Whether to save results
        save_plots (bool): Whether to save plots
        
    Returns:
        dict: All comparison results
    """
    print(f"\n{'='*80}")
    print(f"ALL COMPARISONS: {target} ({optimization_type})")
    print(f"{'='*80}")
    
    comparisons = [
        ('ernie', 'mapped'),
        ('ernie', 'optimized'),
        ('mapped', 'optimized')
    ]
    
    all_results = {}
    
    for condition_a, condition_b in comparisons:
        try:
            results = run_comparison_analysis(
                target, optimization_type, condition_a, condition_b,
                save_results, save_plots
            )
            comparison_name = f"{condition_a}_vs_{condition_b}"
            all_results[comparison_name] = results
        except Exception as e:
            print(f"Error in comparison {condition_a} vs {condition_b}: {e}")
            all_results[f"{condition_a}_vs_{condition_b}"] = None
    
    # Save combined results
    if save_results:
        try:
            results_dir = Path(__file__).parent.parent / 'results' / 'tables'
            results_dir.mkdir(parents=True, exist_ok=True)
            
            combined_data = {
                'analysis_info': {
                    'analysis_type': 'all_comparisons',
                    'target': target,
                    'optimization_type': optimization_type,
                    'timestamp': datetime.now().isoformat()
                },
                'comparisons': all_results
            }
            
            json_filename = f"all_comparisons_{target}_{optimization_type}_results.json"
            json_filepath = results_dir / json_filename
            with open(json_filepath, 'w') as f:
                json.dump(combined_data, f, indent=2, default=str)
            print(f"\n✓ Combined results saved: {json_filename}")
            
        except Exception as e:
            print(f"Error saving combined results: {e}")
    
    return all_results


def run_all_comparisons_all_targets(save_results: bool = True, save_plots: bool = True) -> dict:
    """
    Run all comparisons for all targets and optimization types.
    
    Args:
        save_results (bool): Whether to save results
        save_plots (bool): Whether to save plots
        
    Returns:
        dict: All results for all targets
    """
    print("Running all comparisons for all available combinations...")
    
    targets = ['Left_Insula', 'Right_Hippocampus']
    optimization_types = ['max', 'normal']
    
    all_results = {}
    
    for target in targets:
        all_results[target] = {}
        
        for opt_type in optimization_types:
            print(f"\nProcessing: {target} - {opt_type}")
            
            try:
                results = run_all_comparisons_for_target(target, opt_type, save_results, save_plots)
                all_results[target][opt_type] = results
            except Exception as e:
                print(f"Error processing {target} - {opt_type}: {e}")
                all_results[target][opt_type] = None
    
    return all_results


def main():
    """
    Main function to run comparison analyses.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Comprehensive Comparison Analysis')
    parser.add_argument('--target', type=str, 
                                               choices=['Left_Insula', 'Right_Hippocampus', 'sphere_x-36.1_y14.14_z0.33'],
                       help='Target region to analyze')
    parser.add_argument('--optimization', type=str, 
                       choices=['max', 'normal'],
                       help='Optimization type to analyze')
    parser.add_argument('--condition-a', type=str,
                       choices=['ernie', 'optimized', 'mapped'],
                       help='First condition for comparison')
    parser.add_argument('--condition-b', type=str,
                       choices=['ernie', 'optimized', 'mapped'],
                       help='Second condition for comparison')
    parser.add_argument('--all-targets', action='store_true',
                       help='Run all comparisons for all targets')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results and plots')
    
    args = parser.parse_args()
    
    if args.all_targets:
        run_all_comparisons_all_targets(
            save_results=not args.no_save, 
            save_plots=not args.no_save
        )
    elif args.target and args.optimization and args.condition_a and args.condition_b:
        run_comparison_analysis(
            args.target, args.optimization, args.condition_a, args.condition_b,
            save_results=not args.no_save, 
            save_plots=not args.no_save
        )
    elif args.target and args.optimization:
        run_all_comparisons_for_target(
            args.target, args.optimization,
            save_results=not args.no_save, 
            save_plots=not args.no_save
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 