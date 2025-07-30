"""
Visualization Module for TI-Toolbox Research

This module provides publication-ready plotting functions for the three main research questions:
- Q1: Individualization effects (Individual vs Generalized models)
- Q2: Mapping effects (Free vs Mapped optimization)  
- Q3: Demographic factors and individual variability

Author: TI-Toolbox Research Team
Date: July 2024
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('default')
sns.set_palette("husl")


class PublicationPlotter:
    """
    A class to create publication-ready plots for TI-Toolbox research.
    
    This class provides methods for creating consistent, high-quality
    visualizations for all research questions.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 150):
        """
        Initialize the PublicationPlotter.
        
        Args:
            figsize (Tuple[int, int]): Default figure size (width, height)
            dpi (int): Figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'condition_a': '#2E86AB',  # Blue
            'condition_b': '#A23B72',  # Red
            'correlation': '#F18F01',  # Orange
            'regression': '#C73E1D'    # Dark red
        }
    
    def setup_publication_style(self):
        """Set up Nature publication-quality plotting style."""
        plt.rcParams.update({
            'font.size': 8,
            'axes.titlesize': 9,
            'axes.labelsize': 8,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'figure.titlesize': 10,
            'lines.linewidth': 1,
            'axes.linewidth': 0.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'font.family': 'Arial',
            'mathtext.fontset': 'custom',
            'mathtext.rm': 'Arial',
            'mathtext.it': 'Arial:italic',
            'mathtext.bf': 'Arial:bold'
        })
    
    def create_paired_comparison_plot(self, df: pd.DataFrame, 
                                    variables: List[str] = None,
                                    title: str = "Paired Comparison Results",
                                    stats_results: Dict = None) -> plt.Figure:
        """
        Create publication-ready paired comparison plots for Q1 and Q2.
        
        Args:
            df (pd.DataFrame): Dataframe with 'condition' column and variables
            variables (List[str]): Variables to plot (default: ['ROI_Mean', 'ROI_Max', 'ROI_Focality'])
            title (str): Plot title
            stats_results (Dict): Statistical results from paired_comparison_analysis
            
        Returns:
            plt.Figure: Publication-ready figure
        """
        if variables is None:
            variables = ['ROI_Mean', 'ROI_Max', 'ROI_Focality', 'Normal_Mean', 'Normal_Max']
        
        # Filter to only include variables that exist in the dataframe
        variables = [var for var in variables if var in df.columns]
        
        self.setup_publication_style()
        fig, axes = plt.subplots(1, len(variables), figsize=(2.5*len(variables), 2.0))
        
        if len(variables) == 1:
            axes = [axes]
        
        for i, var in enumerate(variables):
            ax = axes[i]
            
            # Get paired data for each participant
            pivot_df = df.pivot(index='Subject_ID', columns='condition', values=var).dropna()
            
            # Calculate means and standard errors
            condition_names = pivot_df.columns
            means = pivot_df.mean()
            sems = pivot_df.sem()
            
            # Create bar plot for means
            x_pos = np.arange(len(condition_names))
            bars = ax.bar(x_pos, means, yerr=sems, capsize=3, 
                         color=[self.colors['condition_a'], self.colors['condition_b']],
                         alpha=0.8, width=0.6)
            
            # Add individual data points
            for j, condition in enumerate(condition_names):
                ax.scatter([j] * len(pivot_df), pivot_df[condition], 
                          color='black', alpha=0.4, s=15, zorder=3)
            
            # Add connecting lines for paired data
            for _, row in pivot_df.iterrows():
                ax.plot([0, 1], [row.iloc[0], row.iloc[1]], 
                       color='gray', alpha=0.2, linewidth=0.5)
            
            # Add group average lines
            for j, condition in enumerate(condition_names):
                mean_val = means[j]
                ax.axhline(y=mean_val, xmin=j-0.3, xmax=j+0.3, 
                          color=self.colors['condition_a' if j == 0 else 'condition_b'], 
                          linestyle='--', linewidth=1, alpha=0.8)
                # Add mean value text
                ax.text(j, mean_val + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02, 
                       f'{mean_val:.3f}', ha='center', va='bottom', 
                       fontsize=6, fontweight='bold', 
                       color=self.colors['condition_a' if j == 0 else 'condition_b'])
            
            # Add statistical information if available
            if stats_results and var in stats_results:
                stats = stats_results[var]
                p_val = stats['p_value']
                cohens_d = stats['cohens_d']
                percent_change = stats['percent_change']
                
                # Significance indicator
                if p_val < 0.001:
                    sig_text = "***"
                elif p_val < 0.01:
                    sig_text = "**"
                elif p_val < 0.05:
                    sig_text = "*"
                else:
                    sig_text = "ns"
                
                # Add statistical text box
                stats_text = f"p = {p_val:.3f}\nd = {cohens_d:.2f}\nΔ% = {percent_change:.1f}%\n{sig_text}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=6,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=2))
            
            # Customize plot
            ax.set_xlabel('Condition')
            ax.set_ylabel(f'{var} (V/m)')
            ax.set_title(f'{var}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(condition_names)
            ax.grid(True, alpha=0.2)
        
        fig.suptitle(title, fontsize=9, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_correlation_plot(self, df: pd.DataFrame, 
                              target_vars: List[str], 
                              predictor_vars: List[str],
                              title: str = "Correlation Analysis") -> plt.Figure:
        """
        Create publication-ready correlation plots for Q3.
        
        Args:
            df (pd.DataFrame): Dataframe with target and predictor variables
            target_vars (List[str]): Target variables
            predictor_vars (List[str]): Predictor variables
            title (str): Plot title
            
        Returns:
            plt.Figure: Publication-ready figure
        """
        self.setup_publication_style()
        
        # Calculate number of subplots needed
        n_targets = len([var for var in target_vars if var in df.columns])
        n_predictors = len([var for var in predictor_vars if var in df.columns])
        
        if n_targets == 0 or n_predictors == 0:
            raise ValueError("No valid target or predictor variables found")
        
        fig, axes = plt.subplots(n_targets, n_predictors, 
                                figsize=(4*n_predictors, 4*n_targets))
        
        if n_targets == 1 and n_predictors == 1:
            axes = np.array([[axes]])
        elif n_targets == 1:
            axes = axes.reshape(1, -1)
        elif n_predictors == 1:
            axes = axes.reshape(-1, 1)
        
        plot_count = 0
        
        for i, target in enumerate(target_vars):
            if target not in df.columns:
                continue
                
            for j, predictor in enumerate(predictor_vars):
                if predictor not in df.columns:
                    continue
                
                ax = axes[i, j]
                
                # Remove missing values
                valid_data = df[[target, predictor]].dropna()
                
                if len(valid_data) < 3:
                    ax.text(0.5, 0.5, 'Insufficient data', 
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Create scatter plot
                ax.scatter(valid_data[predictor], valid_data[target], 
                          alpha=0.7, s=60, color=self.colors['correlation'])
                
                # Add trend line
                z = np.polyfit(valid_data[predictor], valid_data[target], 1)
                p = np.poly1d(z)
                ax.plot(valid_data[predictor], p(valid_data[predictor]), 
                       "r--", alpha=0.8, linewidth=2)
                
                # Calculate and display correlation
                correlation, p_value = np.corrcoef(valid_data[predictor], valid_data[target])[0, 1], 0
                try:
                    from scipy.stats import pearsonr
                    correlation, p_value = pearsonr(valid_data[predictor], valid_data[target])
                except:
                    pass
                
                # Add correlation text
                sig_text = "**" if p_value < 0.05 else ""
                ax.text(0.05, 0.95, f'r = {correlation:.3f}{sig_text}\np = {p_value:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Customize plot
                ax.set_xlabel(predictor.replace('_', ' ').title())
                ax.set_ylabel(target.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
                
                plot_count += 1
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_regression_plot(self, df: pd.DataFrame, 
                             target_var: str, 
                             predictor_vars: List[str],
                             title: str = "Multiple Regression Analysis") -> plt.Figure:
        """
        Create publication-ready regression plots for Q3.
        
        Args:
            df (pd.DataFrame): Dataframe with target and predictor variables
            target_var (str): Target variable
            predictor_vars (List[str]): Predictor variables
            title (str): Plot title
            
        Returns:
            plt.Figure: Publication-ready figure
        """
        self.setup_publication_style()
        
        # Prepare data
        valid_predictors = [var for var in predictor_vars if var in df.columns]
        
        if target_var not in df.columns or not valid_predictors:
            raise ValueError("Invalid target or predictor variables")
        
        # Remove missing values
        analysis_data = df[[target_var] + valid_predictors].dropna()
        
        if len(analysis_data) < len(valid_predictors) + 2:
            raise ValueError("Insufficient data for regression analysis")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        
        # Panel A: Actual vs Predicted
        ax0 = fig.add_subplot(gs[0, 0])
        
        # Fit regression model
        from statsmodels.api import OLS, add_constant
        X = add_constant(analysis_data[valid_predictors])
        y = analysis_data[target_var]
        model = OLS(y, X).fit()
        y_pred = model.predict(X)
        
        # Plot actual vs predicted
        ax0.scatter(y, y_pred, alpha=0.7, s=40, color='tab:blue')
        ax0.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', alpha=0.8, linewidth=1.5)
        ax0.set_xlabel('Actual Values (V/m)')
        ax0.set_ylabel('Predicted Values (V/m)')
        ax0.set_title(f'A. Actual vs Predicted\nR² = {model.rsquared:.3f}')
        ax0.grid(True, alpha=0.2)
        
        # Panel B: Residuals vs Predicted
        ax1 = fig.add_subplot(gs[0, 1])
        residuals = y - y_pred
        ax1.scatter(y_pred, residuals, alpha=0.7, s=40, color='tab:green')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.8, linewidth=1.5)
        ax1.set_xlabel('Predicted Values (V/m)')
        ax1.set_ylabel('Residuals (V/m)')
        ax1.set_title('B. Residuals vs Predicted')
        ax1.grid(True, alpha=0.2)
        
        # Panel C: Coefficient plot
        ax2 = fig.add_subplot(gs[1, :])
        
        # Extract coefficients and p-values
        coefs = model.params[1:]  # Exclude intercept
        p_vals = model.pvalues[1:]
        
        # Create coefficient plot
        y_pos = np.arange(len(coefs))
        colors = ['tab:red' if p < 0.05 else 'tab:gray' for p in p_vals]
        
        bars = ax2.barh(y_pos, coefs, color=colors, alpha=0.7, height=0.6)
        
        # Add significance indicators and values
        for i, (coef, p_val) in enumerate(zip(coefs, p_vals)):
            sig_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            p_text = f"p = {p_val:.3f}"
            
            # Position text based on coefficient sign
            if coef >= 0:
                ax2.text(coef + 0.0001, i, f"{sig_text}\n{p_text}", 
                        va='center', ha='left', fontsize=8)
            else:
                ax2.text(coef - 0.0001, i, f"{sig_text}\n{p_text}", 
                        va='center', ha='right', fontsize=8)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([var.replace('_', ' ').title() for var in valid_predictors])
        ax2.set_xlabel('Coefficient Value')
        ax2.set_title('C. Regression Coefficients')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax2.grid(True, alpha=0.2)
        
        # Add model summary text
        model_text = f"Model: R² = {model.rsquared:.3f}, F = {model.fvalue:.2f}, p = {model.f_pvalue:.3f}"
        fig.text(0.5, 0.02, model_text, ha='center', fontsize=10, style='italic')
        
        fig.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        return fig
    
    def create_summary_plot(self, results: Dict, analysis_type: str,
                          target: str, optimization_type: str) -> plt.Figure:
        """
        Create a summary plot showing key results for any analysis type.
        
        Args:
            results (Dict): Analysis results
            analysis_type (str): Type of analysis ('Q1', 'Q2', or 'Q3')
            target (str): Target region
            optimization_type (str): Optimization type
            
        Returns:
            plt.Figure: Summary figure
        """
        self.setup_publication_style()
        
        if analysis_type in ['Q1', 'Q2']:
            return self._create_comparison_summary_plot(results, analysis_type, target, optimization_type)
        elif analysis_type == 'Q3':
            return self._create_demographics_summary_plot(results, target, optimization_type)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    def _create_comparison_summary_plot(self, results: Dict, analysis_type: str,
                                      target: str, optimization_type: str) -> plt.Figure:
        """Create summary plot for Q1 and Q2 analyses."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        variables = list(results['comparison_results'].keys())
        p_values = [results['comparison_results'][var]['p_value'] for var in variables]
        effect_sizes = [results['comparison_results'][var]['cohens_d'] for var in variables]
        significant = [results['comparison_results'][var]['significant'] for var in variables]
        
        # Plot 1: P-values
        colors = ['red' if sig else 'gray' for sig in significant]
        bars1 = ax1.bar(variables, p_values, color=colors, alpha=0.7)
        ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, label='α = 0.05')
        ax1.set_ylabel('P-value')
        ax1.set_title('Statistical Significance')
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Effect sizes
        colors = ['red' if sig else 'gray' for sig in significant]
        bars2 = ax2.bar(variables, effect_sizes, color=colors, alpha=0.7)
        ax2.set_ylabel("Cohen's d")
        ax2.set_title('Effect Sizes')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add significance indicators
        for i, sig in enumerate(significant):
            if sig:
                ax1.text(i, p_values[i] + 0.02, '*', ha='center', va='bottom', fontsize=16)
                ax2.text(i, effect_sizes[i] + (0.02 if effect_sizes[i] >= 0 else -0.02), '*', 
                        ha='center', va='bottom' if effect_sizes[i] >= 0 else 'top', fontsize=16)
        
        fig.suptitle(f'{analysis_type} Analysis: {target} ({optimization_type})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _create_demographics_summary_plot(self, results: Dict, 
                                        target: str, optimization_type: str) -> plt.Figure:
        """Create summary plot for Q3 analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract correlation data
        if 'correlation_results' in results:
            corr_data = results['correlation_results']
            
            # Prepare data for heatmap
            target_vars = list(corr_data.keys())
            predictor_vars = list(corr_data[target_vars[0]].keys()) if target_vars else []
            
            corr_matrix = np.zeros((len(target_vars), len(predictor_vars)))
            p_matrix = np.zeros((len(target_vars), len(predictor_vars)))
            
            for i, target_var in enumerate(target_vars):
                for j, predictor in enumerate(predictor_vars):
                    if predictor in corr_data[target_var]:
                        corr_matrix[i, j] = corr_data[target_var][predictor]['correlation']
                        p_matrix[i, j] = corr_data[target_var][predictor]['p_value']
            
            # Plot 1: Correlation heatmap
            im1 = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax1.set_xticks(range(len(predictor_vars)))
            ax1.set_yticks(range(len(target_vars)))
            ax1.set_xticklabels([var.replace('_', ' ').title() for var in predictor_vars], rotation=45)
            ax1.set_yticklabels([var.replace('_', ' ').title() for var in target_vars])
            ax1.set_title('Correlation Matrix')
            
            # Add correlation values
            for i in range(len(target_vars)):
                for j in range(len(predictor_vars)):
                    text = ax1.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Regression results
        if 'regression_results' in results and results['regression_results']:
            reg_results = results['regression_results']
            
            # Extract coefficients and p-values
            coefs = list(reg_results['coefficients'].values())[1:]  # Exclude intercept
            p_vals = list(reg_results['p_values'].values())[1:]
            var_names = list(reg_results['coefficients'].keys())[1:]
            
            colors = ['red' if p < 0.05 else 'gray' for p in p_vals]
            bars = ax2.bar(range(len(coefs)), coefs, color=colors, alpha=0.7)
            
            ax2.set_xticks(range(len(coefs)))
            ax2.set_xticklabels([var.replace('_', ' ').title() for var in var_names], rotation=45)
            ax2.set_ylabel('Coefficient Value')
            ax2.set_title(f'Regression Coefficients\nR² = {reg_results["r_squared"]:.3f}')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f'Q3 Analysis: {target} ({optimization_type})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig


def create_q1_plots(df: pd.DataFrame, target: str, optimization_type: str,
                   save_path: str = None, stats_results: Dict = None) -> List[plt.Figure]:
    """
    Create all plots for Q1 (Individualization) analysis.
    
    Args:
        df (pd.DataFrame): Comparison dataset
        target (str): Target region
        optimization_type (str): Optimization type
        save_path (str): Path to save figures (optional)
        stats_results (Dict): Statistical results from analysis
        
    Returns:
        List[plt.Figure]: List of created figures
    """
    plotter = PublicationPlotter()
    figures = []
    
    # Create paired comparison plot
    fig1 = plotter.create_paired_comparison_plot(
        df, title=f"Q1: Individualization Effects - {target} ({optimization_type})",
        stats_results=stats_results
    )
    figures.append(fig1)
    
    if save_path:
        fig1.savefig(f"{save_path}/q1_paired_comparison_{target}_{optimization_type}.png", 
                    dpi=300, bbox_inches='tight')
    
    return figures


def create_q2_plots(df: pd.DataFrame, target: str, optimization_type: str,
                   save_path: str = None, stats_results: Dict = None) -> List[plt.Figure]:
    """
    Create all plots for Q2 (Mapping) analysis.
    
    Args:
        df (pd.DataFrame): Comparison dataset
        target (str): Target region
        optimization_type (str): Optimization type
        save_path (str): Path to save figures (optional)
        stats_results (Dict): Statistical results from analysis
        
    Returns:
        List[plt.Figure]: List of created figures
    """
    plotter = PublicationPlotter()
    figures = []
    
    # Create paired comparison plot
    fig1 = plotter.create_paired_comparison_plot(
        df, title=f"Q2: Mapping Effects - {target} ({optimization_type})",
        stats_results=stats_results
    )
    figures.append(fig1)
    
    if save_path:
        fig1.savefig(f"{save_path}/q2_paired_comparison_{target}_{optimization_type}.png", 
                    dpi=300, bbox_inches='tight')
    
    return figures


def create_q3_plots(df: pd.DataFrame, target: str, optimization_type: str,
                   save_path: str = None) -> List[plt.Figure]:
    """
    Create all plots for Q3 (Demographics) analysis.
    
    Args:
        df (pd.DataFrame): Dataset with demographics
        target (str): Target region
        optimization_type (str): Optimization type
        save_path (str): Path to save figures (optional)
        
    Returns:
        List[plt.Figure]: List of created figures
    """
    plotter = PublicationPlotter()
    figures = []
    
    # Define variables
    target_vars = ['ROI_Mean', 'ROI_Max', 'Normal_Mean', 'Normal_Max']
    predictor_vars = ['age', 'bone_volume', 'bone_mean_thick']  # Demographics columns
    
    # Create correlation plot
    fig1 = plotter.create_correlation_plot(
        df, target_vars, predictor_vars,
        title=f"Q3: Demographic Correlations - {target} ({optimization_type})"
    )
    figures.append(fig1)
    
    # Create regression plot (only if we have sufficient data)
    try:
        if len(df.dropna(subset=predictor_vars)) >= len(predictor_vars) + 2:
            fig2 = plotter.create_regression_plot(
                df, 'ROI_Mean', predictor_vars,
                title=f"Q3: Multiple Regression - {target} ({optimization_type})"
            )
            figures.append(fig2)
            
            if save_path:
                fig2.savefig(f"{save_path}/q3_regression_{target}_{optimization_type}.png", 
                            dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"   Note: Regression plot not created due to insufficient data: {e}")
    
    if save_path:
        fig1.savefig(f"{save_path}/q3_correlations_{target}_{optimization_type}.png", 
                    dpi=300, bbox_inches='tight')
    
    return figures


if __name__ == "__main__":
    # Example usage and testing
    print("Visualization Module for TI-Toolbox Research")
    print("Available functions:")
    print("- create_q1_plots()")
    print("- create_q2_plots()")
    print("- create_q3_plots()")
    print("- PublicationPlotter class") 