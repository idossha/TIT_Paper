#!/usr/bin/env python3
"""
Streamlined Analysis Runner for TI-Toolbox Research

This module provides a unified interface for running TI-Toolbox analyses using
configuration files. It handles all the complexity of running different research
questions across multiple regions and optimization types.

Author: TI-Toolbox Research Team
Date: July 2024
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging

# Add analysis directory to path
sys.path.append(str(Path(__file__).parent))

from config import ConfigLoader
from data import DataLoader, load_q3_data, load_q3_data_enhanced, load_comparison_data
from stats import StatisticalAnalyzer, analyze_q3_demographics, analyze_pairwise_comparisons
from plots import PublicationPlotter, create_q3_plots, create_q3_plots_enhanced, create_q1_plots, create_q2_plots


class AnalysisRunner:
    """
    A streamlined analysis runner that uses configuration files to manage
    all analysis parameters and execution.
    """
    
    def __init__(self, config_file: str = "settings.yaml"):
        """
        Initialize the AnalysisRunner.
        
        Args:
            config_file (str): Path to configuration file
        """
        self.config = ConfigLoader(config_file)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_loader = DataLoader(str(self.config.get_data_dir()))
        self.statistical_analyzer = StatisticalAnalyzer()
        self.plotter = PublicationPlotter()
        
        # Create output directories
        self._create_output_directories()
    
    def _create_output_directories(self):
        """Create output directories if they don't exist."""
        output_dirs = self.config.get_output_dirs()
        
        for dir_name, dir_path in output_dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created output directory: {dir_path}")
    
    def run_single_analysis(self, region: str, optimization_type: str, 
                           questions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run analysis for a single region and optimization type.
        
        Args:
            region (str): Target region name
            optimization_type (str): Optimization type
            questions (List[str], optional): Specific questions to run
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        if questions is None:
            questions = self.config.get_questions()
        
        self.logger.info(f"Running analysis for {region} ({optimization_type})")
        self.logger.info(f"Research questions: {', '.join(questions)}")
        
        results = {
            'analysis_info': {
                'region': region,
                'optimization_type': optimization_type,
                'questions': questions,
                'timestamp': datetime.now().isoformat(),
                'config_file': str(self.config.config_file)
            },
            'results': {}
        }
        
        # Run each research question
        for question in questions:
            try:
                if question == 'Q3':
                    question_results = self._run_q3_analysis(region, optimization_type)
                elif question == 'pairwise':
                    question_results = self._run_comparison_analysis(region, optimization_type)
                else:
                    self.logger.warning(f"Unknown research question: {question}")
                    continue
                
                results['results'][question] = question_results
                self.logger.info(f"✓ Completed {question} analysis")
                
            except Exception as e:
                self.logger.error(f"✗ Failed {question} analysis: {e}")
                results['results'][question] = None
        
        # Save results
        self._save_results(results, region, optimization_type)
        
        return results
    
    def _run_q3_analysis(self, region: str, optimization_type: str) -> Dict[str, Any]:
        """Run comprehensive Q3 demographic analysis with all three conditions."""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE Q3 ANALYSIS: Demographic Factors")
        print(f"Target: {region}")
        print(f"Optimization: {optimization_type}")
        print(f"{'='*60}")
        
        # Load enhanced data (all three conditions)
        print("\n1. Loading enhanced data...")
        df, quality_report = load_q3_data_enhanced(region, optimization_type)
        print(f"   ✓ Enhanced data loaded successfully: {len(df)} total participants")
        print(f"   ✓ Data quality: {quality_report['n_participants']} participants, {sum(quality_report['missing_values'].values())} missing values")
        
        # Perform statistical analysis
        print("\n2. Performing statistical analysis...")
        analysis_results = analyze_q3_demographics(df, region, optimization_type)
        print("   ✓ Statistical analysis completed")
        
        # Create enhanced visualizations
        print("\n3. Creating enhanced visualizations...")
        output_dirs = self.config.get_output_dirs()
        figures_dir = output_dirs['figures']
        figures = create_q3_plots_enhanced(df, region, optimization_type, str(figures_dir))
        print(f"   ✓ Created {len(figures)} figure(s)")
        
        # Save results
        print("\n4. Saving results...")
        self._save_q3_results_enhanced(analysis_results, region, optimization_type)
        
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE Q3 ANALYSIS COMPLETED")
        print(f"{'='*60}")
        
        return analysis_results
    
    def _run_comparison_analysis(self, region: str, optimization_type: str) -> Dict[str, Any]:
        """Run comparison analysis."""
        print(f"\n{'='*80}")
        print(f"ALL COMPARISONS: {region} ({optimization_type})")
        print(f"{'='*80}")
        
        all_results = {}
        comparisons = [
            ('ernie', 'mapped'),
            ('ernie', 'optimized'),
            ('mapped', 'optimized')
        ]
        
        for condition_a, condition_b in comparisons:
            comparison_name = f"{condition_a}_vs_{condition_b}"
            print(f"\n{'='*60}")
            print(f"COMPARISON: {comparison_name.upper()}")
            print(f"Target: {region}")
            print(f"Optimization: {optimization_type}")
            print(f"{'='*60}")
            
            # Load data
            print("\n1. Loading data...")
            df, quality_report = load_comparison_data(region, optimization_type, condition_a, condition_b)
            print(f"   ✓ Data loaded successfully: {len(df)} participants")
            print(f"   ✓ Data quality: {quality_report['n_participants']} participants, {sum(quality_report['missing_values'].values())} missing values")
            
            # Perform statistical analysis
            print("\n2. Performing statistical analysis...")
            analysis_results = analyze_pairwise_comparisons(df, region, optimization_type, condition_a, condition_b)
            print("   ✓ Statistical analysis completed")
            
            # Create visualizations
            print("\n3. Creating visualizations...")
            output_dirs = self.config.get_output_dirs()
            figures_dir = output_dirs['figures']
            
            # Create comparison plots
            plotter = PublicationPlotter()
            fig1 = plotter.create_paired_comparison_plot(
                df, title=f"Comparison: {condition_a} vs {condition_b} - {region} ({optimization_type})",
                stats_results=analysis_results['comparison_results'], optimization_type=optimization_type
            )
            fig2 = plotter.create_normality_check_plot(
                df, title=f"Normality Check: {condition_a} vs {condition_b} - {region} ({optimization_type})",
                stats_results=analysis_results['comparison_results'], optimization_type=optimization_type
            )
            
            # Save figures
            fig1.savefig(figures_dir / f"comparison_{comparison_name}_{region}_{optimization_type}.png", 
                        dpi=300, bbox_inches='tight', facecolor='white')
            fig2.savefig(figures_dir / f"normality_{comparison_name}_{region}_{optimization_type}.png", 
                        dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   ✓ Created 2 figures")
            print(f"   ✓ Saved: comparison_{comparison_name}_{region}_{optimization_type}.png")
            print(f"   ✓ Saved: normality_{comparison_name}_{region}_{optimization_type}.png")
            
            # Save results
            print("\n4. Saving results...")
            self._save_comparison_results(analysis_results, region, optimization_type, condition_a, condition_b)
            
            print(f"\n{'='*60}")
            print(f"COMPARISON {comparison_name.upper()} COMPLETED")
            print(f"{'='*60}")
            
            all_results[comparison_name] = analysis_results
        
        # Save combined results
        self._save_combined_comparison_results(all_results, region, optimization_type)
        print(f"\n✓ Combined results saved: all_comparisons_{region}_{optimization_type}_results.json")
        
        return all_results
    
    def run_all_analyses(self) -> Dict[str, Any]:
        """
        Run analysis for all regions and optimization types defined in configuration.
        
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        self.logger.info("Running analysis for all regions and optimization types")
        
        all_results = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'config_file': str(self.config.config_file),
                'mode': 'all_regions'
            },
            'results': {}
        }
        
        region_combinations = self.config.get_region_combinations()
        
        for region, optimization_type in region_combinations:
            try:
                region_results = self.run_single_analysis(region, optimization_type)
                all_results['results'][f"{region}_{optimization_type}"] = region_results
                
            except Exception as e:
                self.logger.error(f"Failed analysis for {region} ({optimization_type}): {e}")
                all_results['results'][f"{region}_{optimization_type}"] = None
        
        # Save comprehensive results
        self._save_comprehensive_results(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Any], region: str, optimization_type: str):
        """Save analysis results to file."""
        analysis_settings = self.config.get_analysis_settings()
        if not analysis_settings.get('save_results', True):
            return
        
        output_dirs = self.config.get_output_dirs()
        results_dir = output_dirs['results']
        
        filename = f"complete_analysis_{region}_{optimization_type}.json"
        filepath = results_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Saved results to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Save comprehensive analysis results."""
        analysis_settings = self.config.get_analysis_settings()
        if not analysis_settings.get('save_results', True):
            return
        
        output_dirs = self.config.get_output_dirs()
        results_dir = output_dirs['results']
        
        filename = "comprehensive_analysis_summary.json"
        filepath = results_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Saved comprehensive results to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save comprehensive results: {e}")
    
    def validate_data_availability(self) -> bool:
        """
        Validate that all required data files are available.
        
        Returns:
            bool: True if all required files are available
        """
        self.logger.info("Validating data availability...")
        
        data_dir = self.config.get_data_dir()
        if not data_dir.exists():
            self.logger.error(f"Data directory not found: {data_dir}")
            return False
        
        region_combinations = self.config.get_region_combinations()
        conditions = self.config.get_conditions()
        
        missing_files = []
        
        for region, optimization_type in region_combinations:
            for condition in conditions:
                # Map condition names to file naming scheme
                condition_map = {
                    'ernie': 'ernie',
                    'mapped': 'mapped',
                    'optimized': 'opt',
                    'opt': 'opt'
                }
                
                file_condition = condition_map.get(condition, condition)
                filename = f"{region}_{optimization_type}_{file_condition}.csv"
                filepath = data_dir / filename
                
                if not filepath.exists():
                    missing_files.append(filename)
        
        if missing_files:
            self.logger.error(f"Missing data files: {missing_files}")
            return False
        
        self.logger.info("✓ All required data files are available")
        return True
    
    def print_analysis_summary(self, results: Dict[str, Any]):
        """Print a summary of analysis results."""
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        analysis_info = results.get('analysis_info', {})
        print(f"Timestamp: {analysis_info.get('timestamp', 'N/A')}")
        print(f"Config file: {analysis_info.get('config_file', 'N/A')}")
        
        if 'region' in analysis_info:
            print(f"Region: {analysis_info['region']}")
            print(f"Optimization: {analysis_info['optimization_type']}")
        
        print(f"\nResearch Questions: {', '.join(analysis_info.get('questions', []))}")
        
        results_data = results.get('results', {})
        print(f"\nResults Summary:")
        
        for question, result in results_data.items():
            if result is not None:
                print(f"  ✓ {question}: Completed")
            else:
                print(f"  ✗ {question}: Failed")
        
        print("="*80)

    def _save_q3_results_enhanced(self, results: Dict[str, Any], region: str, optimization_type: str):
        """Save comprehensive Q3 analysis results."""
        output_dirs = self.config.get_output_dirs()
        results_dir = output_dirs['results']
        
        # Save detailed results
        filename = f"q3_comprehensive_{region}_{optimization_type}_results.json"
        filepath = results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"   ✓ Saved: {filename}")
        
        # Save summary
        summary_filename = f"q3_comprehensive_{region}_{optimization_type}_summary.txt"
        summary_filepath = results_dir / summary_filename
        with open(summary_filepath, 'w') as f:
            f.write(results['summary_report'])
        print(f"   ✓ Saved: {summary_filename}")
    
    def _save_q3_results(self, results: Dict[str, Any], region: str, optimization_type: str):
        """Save Q3 analysis results (legacy method)."""
        output_dirs = self.config.get_output_dirs()
        results_dir = output_dirs['results']
        
        # Save detailed results
        filename = f"q3_{region}_{optimization_type}_results.json"
        filepath = results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"   ✓ Saved: {filename}")
        
        # Save summary
        summary_filename = f"q3_{region}_{optimization_type}_summary.txt"
        summary_filepath = results_dir / summary_filename
        with open(summary_filepath, 'w') as f:
            f.write(results['summary_report'])
        print(f"   ✓ Saved: {summary_filename}")
    
    def _save_comparison_results(self, results: Dict[str, Any], region: str, 
                               optimization_type: str, condition_a: str, condition_b: str):
        """Save individual comparison results."""
        output_dirs = self.config.get_output_dirs()
        results_dir = output_dirs['results']
        
        comparison_name = f"{condition_a}_vs_{condition_b}"
        
        # Save detailed results
        filename = f"comparison_{comparison_name}_{region}_{optimization_type}_results.json"
        filepath = results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"   ✓ Saved: {filename}")
        
        # Save summary
        summary_filename = f"comparison_{comparison_name}_{region}_{optimization_type}_summary.txt"
        summary_filepath = results_dir / summary_filename
        with open(summary_filepath, 'w') as f:
            f.write(results['summary_report'])
        print(f"   ✓ Saved: {summary_filename}")
    
    def _save_combined_comparison_results(self, all_results: Dict[str, Any], region: str, optimization_type: str):
        """Save combined comparison results."""
        output_dirs = self.config.get_output_dirs()
        results_dir = output_dirs['results']
        
        filename = f"all_comparisons_{region}_{optimization_type}_results.json"
        filepath = results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)


def main():
    """Main entry point for the analysis runner."""
    parser = argparse.ArgumentParser(
        description="TI-Toolbox Analysis Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis with default settings
  python main.py
  
  # Run analysis for specific region and optimization type
  python main.py --region Left_Insula --optimization max
  
  # Run all regions and optimization types
  python main.py --all-regions
  
  # Use custom configuration file
  python main.py --config custom_settings.yaml
  
  # Run specific research questions only
  python main.py --questions Q3 pairwise
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='settings.yaml',
        help='Path to configuration file (default: settings.yaml)'
    )
    
    parser.add_argument(
        '--region', '-r',
        help='Target region to analyze'
    )
    
    parser.add_argument(
        '--optimization', '-o',
        help='Optimization type to analyze'
    )
    
    parser.add_argument(
        '--all-regions', '-a',
        action='store_true',
        help='Run analysis for all regions and optimization types'
    )
    
    parser.add_argument(
        '--questions', '-q',
        nargs='+',
        choices=['Q3', 'pairwise'],
        help='Specific research questions to run'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration and data availability'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize analysis runner
        runner = AnalysisRunner(args.config)
        
        # Validate data availability
        if not runner.validate_data_availability():
            print("❌ Data validation failed. Please check your data files.")
            return 1
        
        if args.validate_only:
            print("✅ Configuration and data validation passed.")
            return 0
        
        # Run analysis based on arguments
        if args.all_regions:
            results = runner.run_all_analyses()
        elif args.region and args.optimization:
            results = runner.run_single_analysis(
                args.region, 
                args.optimization, 
                args.questions
            )
        else:
            # Use default settings from configuration
            default_region, default_optimization = runner.config.get_default_region()
            results = runner.run_single_analysis(
                default_region,
                default_optimization,
                args.questions
            )
        
        # Print summary
        runner.print_analysis_summary(results)
        
        print("✅ Analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 