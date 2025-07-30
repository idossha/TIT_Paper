#!/usr/bin/env python3
"""
Test script to demonstrate the new professional plotting capabilities
following Cell/Science/Nature publication standards.

This script creates sample visualizations to showcase the updated styling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotting_utils import PublicationPlotter, PROFESSIONAL_COLORS

def create_sample_data():
    """Create sample data for testing the professional plots."""
    np.random.seed(42)
    n_subjects = 20
    
    # Create sample data for Q1/Q2 comparisons
    comparison_data = []
    for i in range(n_subjects):
        subject_id = f"Subject_{i+1:02d}"
        
        # Q1: Individual vs Generalized
        individual_mean = np.random.normal(0.15, 0.03)
        generalized_mean = individual_mean + np.random.normal(-0.02, 0.01)
        
        comparison_data.extend([
            {'Subject_ID': subject_id, 'condition': 'Individual', 'ROI_Mean': individual_mean, 'ROI_Max': individual_mean * 1.2, 'ROI_Focality': individual_mean * 0.8},
            {'Subject_ID': subject_id, 'condition': 'Generalized', 'ROI_Mean': generalized_mean, 'ROI_Max': generalized_mean * 1.2, 'ROI_Focality': generalized_mean * 0.8}
        ])
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create sample data for Q3 demographics
    demographics_data = []
    for i in range(n_subjects):
        age = np.random.normal(35, 10)
        bone_volume = np.random.normal(1000, 200)
        bone_thickness = np.random.normal(5, 1)
        
        # Correlate ROI values with demographics
        roi_mean = 0.1 + 0.001 * age + 0.0001 * bone_volume + 0.01 * bone_thickness + np.random.normal(0, 0.02)
        
        demographics_data.append({
            'Subject_ID': f"Subject_{i+1:02d}",
            'age': age,
            'bone_volume': bone_volume,
            'bone_mean_thick': bone_thickness,
            'ROI_Mean': roi_mean,
            'ROI_Max': roi_mean * 1.2,
            'Normal_Mean': roi_mean * 0.9,
            'Normal_Max': roi_mean * 1.1
        })
    
    demographics_df = pd.DataFrame(demographics_data)
    
    return comparison_df, demographics_df

def test_professional_plotting():
    """Test the professional plotting capabilities."""
    print("Testing Professional Plotting Capabilities")
    print("=" * 50)
    
    # Create sample data
    comparison_df, demographics_df = create_sample_data()
    
    # Initialize the professional plotter
    plotter = PublicationPlotter()
    
    # Test 1: Paired comparison plot (Q1/Q2 style)
    print("\n1. Creating paired comparison plot...")
    fig1 = plotter.create_paired_comparison_plot(
        comparison_df, 
        variables=['ROI_Mean', 'ROI_Max'],
        title="Professional Paired Comparison Example"
    )
    fig1.savefig('test_paired_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("   ✓ Saved: test_paired_comparison.png")
    
    # Test 2: Correlation plot (Q3 style)
    print("\n2. Creating correlation plot...")
    fig2 = plotter.create_correlation_plot(
        demographics_df,
        target_vars=['ROI_Mean', 'ROI_Max'],
        predictor_vars=['age', 'bone_volume'],
        title="Professional Correlation Analysis Example"
    )
    fig2.savefig('test_correlations.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("   ✓ Saved: test_correlations.png")
    
    # Test 3: Regression plot (Q3 style)
    print("\n3. Creating regression plot...")
    fig3 = plotter.create_regression_plot(
        demographics_df,
        target_var='ROI_Mean',
        predictor_vars=['age', 'bone_volume', 'bone_mean_thick'],
        title="Professional Regression Analysis Example"
    )
    fig3.savefig('test_regression.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("   ✓ Saved: test_regression.png")
    
    # Test 4: Color palette demonstration
    print("\n4. Creating color palette demonstration...")
    fig4, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    plotter.setup_publication_style()
    
    colors = list(PROFESSIONAL_COLORS.values())
    color_names = list(PROFESSIONAL_COLORS.keys())
    
    for i, (color, name) in enumerate(zip(colors, color_names)):
        ax.bar(i, 1, color=color, alpha=0.8, edgecolor=PROFESSIONAL_COLORS['primary'], linewidth=0.5)
        ax.text(i, 0.5, name, ha='center', va='center', fontsize=8, 
               color='white' if i in [1, 2, 3, 4] else PROFESSIONAL_COLORS['primary'],
               fontweight='bold')
    
    ax.set_xlim(-0.5, len(colors) - 0.5)
    ax.set_ylim(0, 1.2)
    ax.set_title('Professional Color Palette', fontsize=10, fontweight='bold', 
                color=PROFESSIONAL_COLORS['primary'])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    fig4.savefig('test_color_palette.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("   ✓ Saved: test_color_palette.png")
    
    print("\n" + "=" * 50)
    print("Professional plotting test completed successfully!")
    print("Generated files:")
    print("- test_paired_comparison.png")
    print("- test_correlations.png") 
    print("- test_regression.png")
    print("- test_color_palette.png")
    print("\nAll figures follow Cell/Science/Nature publication standards.")

if __name__ == "__main__":
    test_professional_plotting() 