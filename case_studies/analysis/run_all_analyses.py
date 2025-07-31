#!/usr/bin/env python3
"""
Master Analysis Script for TI-Toolbox Research

This script provides a unified interface to run all three research question analyses:
- Q1: Individualization effects (Individual vs Generalized models)
- Q2: Mapping effects (Free vs Mapped optimization)
- Q3: Demographic factors and individual variability

Author: TI-Toolbox Research Team
Date: July 2024
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add analysis directory to path
sys.path.append(str(Path(__file__).parent))


def run_q3_analysis(target: str, optimization_type: str, save_results: bool = True):
    """Run Q3 analysis for demographic factors."""
    from run_q3_analysis import run_q3_analysis as run_q3
    return run_q3(target, optimization_type, save_results, save_results)


def run_comprehensive_comparisons(target: str, optimization_type: str, save_results: bool = True):
    """Run comprehensive pairwise comparisons."""
    from run_all_comparisons import run_all_comparisons_for_target
    return run_all_comparisons_for_target(target, optimization_type, save_results)


def run_complete_analysis(target: str, optimization_type: str, 
                         questions: list = None, save_results: bool = True) -> dict:
    """
    Run complete analysis for a specific target and optimization type.
    
    Args:
        target (str): Target region
        optimization_type (str): Optimization type
        questions (list): List of research questions to run (default: Q3, comparisons)
        save_results (bool): Whether to save results
        
    Returns:
        dict: Complete analysis results
    """
    if questions is None:
        questions = ['Q3', 'comparisons']
    
    print(f"\n{'='*80}")
    print(f"COMPLETE ANALYSIS: {target} ({optimization_type})")
    print(f"Research Questions: {', '.join(questions)}")
    print(f"{'='*80}")
    
    results = {
        'analysis_info': {
            'target': target,
            'optimization_type': optimization_type,
            'questions': questions,
            'timestamp': datetime.now().isoformat()
        },
        'results': {}
    }
    
    # Run comprehensive comparisons (replaces Q1/Q2)
    if 'comparisons' in questions:
        print(f"\n{'='*40}")
        print("RUNNING COMPREHENSIVE COMPARISONS")
        print(f"{'='*40}")
        try:
            comparison_results = run_comprehensive_comparisons(target, optimization_type, save_results)
            results['results']['comparisons'] = comparison_results
            print("✓ Comprehensive comparisons completed successfully")
        except Exception as e:
            print(f"✗ Comprehensive comparisons failed: {e}")
            results['results']['comparisons'] = None
    
    # Run Q3 analysis
    if 'Q3' in questions:
        print(f"\n{'='*40}")
        print("RUNNING Q3: Demographic Factors")
        print(f"{'='*40}")
        try:
            q3_results = run_q3_analysis(target, optimization_type, save_results)
            results['results']['Q3'] = q3_results
            print("✓ Q3 analysis completed successfully")
        except Exception as e:
            print(f"✗ Q3 analysis failed: {e}")
            results['results']['Q3'] = None
    
    # Save complete results
    if save_results:
        results_dir = Path(__file__).parent.parent / 'results' / 'tables'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        json_filename = f"complete_analysis_{target}_{optimization_type}.json"
        json_filepath = results_dir / json_filename
        with open(json_filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Complete analysis results saved: {json_filename}")
    
    print(f"\n{'='*80}")
    print("COMPLETE ANALYSIS FINISHED")
    print(f"{'='*80}")
    
    return results


def run_all_targets_and_optimizations(questions: list = None, save_results: bool = True):
    """
    Run analysis for all available targets and optimization types.
    
    Args:
        questions (list): List of research questions to run
        save_results (bool): Whether to save results
    """
    print(f"\n{'='*80}")
    print("RUNNING ALL ANALYSES")
    print(f"Research Questions: {', '.join(questions) if questions else 'All'}")
    print(f"{'='*80}")
    
    # Define all combinations based on experimental design
    combinations = [
        ('Left_Insula', 'max'),
        ('Left_Insula', 'normal'),
        ('Right_Hippocampus', 'max'),
        ('Right_Hippocampus', 'normal'),
        # ('10mm_sphere', 'max'),  # Add when available
        # ('10mm_sphere', 'normal'),  # Add when available
    ]
    
    all_results = {}
    
    for target, opt_type in combinations:
        print(f"\nProcessing: {target} - {opt_type}")
        try:
            results = run_complete_analysis(target, opt_type, questions, save_results)
            all_results[f"{target}_{opt_type}"] = results
        except Exception as e:
            print(f"Error processing {target} - {opt_type}: {e}")
            all_results[f"{target}_{opt_type}"] = None
    
    return all_results


def create_comprehensive_summary():
    """
    Create a comprehensive summary of all analyses.
    """
    print("\nCreating comprehensive summary...")
    
    results_dir = Path(__file__).parent.parent / 'results' / 'tables'
    
    if not results_dir.exists():
        print("No results directory found. Run analyses first.")
        return
    
    # Find all result files
    q3_files = list(results_dir.glob("q3_*_results.json"))
    comparison_files = list(results_dir.glob("comparison_*_results.json"))
    complete_files = list(results_dir.glob("complete_analysis_*.json"))
    
    summary_report = "TI-TOOLBOX RESEARCH - COMPREHENSIVE ANALYSIS SUMMARY\n"
    summary_report += "=" * 80 + "\n\n"
    summary_report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary_report += f"Q3 analyses: {len(q3_files)}\n"
    summary_report += f"Comparison analyses: {len(comparison_files)}\n"
    summary_report += f"Complete analyses: {len(complete_files)}\n\n"
    
    # Summary by analysis type
    for analysis_type, files, name in [("Q3", q3_files, "Demographic Factors"),
                                      ("comparisons", comparison_files, "Pairwise Comparisons")]:
        if files:
            summary_report += f"{analysis_type}: {name}\n"
            summary_report += "-" * 50 + "\n"
            
            for file_path in sorted(files):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    target = data['analysis_info']['target']
                    opt_type = data['analysis_info']['optimization_type']
                    n_participants = data['analysis_info']['n_participants']
                    
                    summary_report += f"  {target} ({opt_type}): {n_participants} participants\n"
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            
            summary_report += "\n"
    
    # Save comprehensive summary
    summary_file = results_dir / "comprehensive_analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary_report)
    
    print(f"✓ Comprehensive summary saved: {summary_file}")
    print("\n" + summary_report)


def main():
    """
    Main function with command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='TI-Toolbox Research Analysis Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all analyses for Left_Insula, max optimization
  python run_all_analyses.py --target Left_Insula --optimization max
  
  # Run only Q1 and Q2 analyses
  python run_all_analyses.py --target Left_Insula --optimization max --questions Q1 Q2
  
  # Run all analyses for all targets
  python run_all_analyses.py --all
  
  # Create comprehensive summary
  python run_all_analyses.py --summary
        """
    )
    
    parser.add_argument('--target', type=str, 
                       choices=['Left_Insula', 'Right_Hippocampus', 'sphere_x-36.1_y14.14_z0.33'],
                       help='Target region to analyze')
    parser.add_argument('--optimization', type=str, 
                       choices=['max', 'normal'],
                       help='Optimization type to analyze')
    parser.add_argument('--questions', nargs='+', 
                       choices=['Q3', 'comparisons'],
                       help='Specific analyses to run (Q3: demographics, comparisons: pairwise comparisons)')
    parser.add_argument('--all', action='store_true',
                       help='Run analysis for all available combinations')
    parser.add_argument('--summary', action='store_true',
                       help='Create comprehensive summary report')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results and plots')
    
    args = parser.parse_args()
    
    save_results = not args.no_save
    
    if args.summary:
        # Create comprehensive summary
        create_comprehensive_summary()
    elif args.all:
        # Run all analyses
        run_all_targets_and_optimizations(args.questions, save_results)
        # Also create summary
        create_comprehensive_summary()
    elif args.target and args.optimization:
        # Run specific analysis
        run_complete_analysis(args.target, args.optimization, args.questions, save_results)
    else:
        # Run default analysis
        print("Running default analysis (Left_Insula, max, Q3 and comparisons)...")
        run_complete_analysis('Left_Insula', 'max', ['Q3', 'comparisons'], save_results)


if __name__ == "__main__":
    main() 