import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr

# Import the updated plotting utilities
from plotting_utils import PublicationPlotter, PROFESSIONAL_COLORS

# --- Load and prepare data ---
bone = pd.read_csv('experiment_0/bone.csv')
hippo = pd.read_csv('experiment_0/opt_hippo.csv')

# Rename mean columns for clarity
hippo = hippo[['Subject_ID', 'mean']].rename(columns={'mean': 'mean_hippo'})

# Merge all on Subject_ID
merged = bone.merge(hippo, on='Subject_ID', how='inner')

# --- Individual correlation analyses ---
# Hippocampus vs Bone Volume
hippo_volume_corr, hippo_volume_p = pearsonr(merged['mean_hippo'], merged['volume'])

# Hippocampus vs Bone Mean Thickness
hippo_thickness_corr, hippo_thickness_p = pearsonr(merged['mean_hippo'], merged['mean'])

# --- Correlation analysis for multiple regression ---
correlation = merged[['mean', 'volume']].corr()

# --- Function to create publication-ready figure for hippocampus ---
def create_publication_figure(filename):
    # Initialize the professional plotter
    plotter = PublicationPlotter()
    plotter.setup_publication_style()
    
    # Multiple regression
    X = add_constant(merged[['mean', 'volume']])
    y = merged['mean_hippo']
    model = sm.OLS(y, X).fit()
    y_pred = model.predict(X)
    r2 = model.rsquared
    pvals = model.pvalues
    coef = model.params
    n = len(y)

    # Prepare summary text
    summary_text = (
        f"Procedure:\n"
        f"- Individual correlations between hippocampus EF and bone characteristics were computed.\n"
        f"- Multiple linear regression was performed to predict electric field (EF) in the hippocampus as a function of bone mean and volume.\n"
        f"\nResults:\n"
        f"- Hippocampus EF vs Bone Volume: r={hippo_volume_corr:.3f}, p={hippo_volume_p:.3f}\n"
        f"- Hippocampus EF vs Bone Thickness: r={hippo_thickness_corr:.3f}, p={hippo_thickness_p:.3f}\n"
        f"- Bone thickness vs volume correlation: r={correlation.iloc[0,1]:.3f}\n"
        f"- Multiple regression R²: {r2:.3f} (n={n})\n"
        f"- Coefficients: Intercept={coef[0]:.3f}, Thickness={coef[1]:.3f} (p={pvals[1]:.3f}), Volume={coef[2]:.3g} (p={pvals[2]:.3f})\n"
        f"- The model explains {r2*100:.1f}% of the variance in hippocampus EF.\n"
    )

    # Create figure with professional styling
    fig = plt.figure(figsize=(16, 10), dpi=300)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2], width_ratios=[1, 1, 1.2],
                         hspace=0.4, wspace=0.4)

    # Panel A: Hippocampus vs Bone Volume
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(merged['volume'], merged['mean_hippo'], alpha=0.7, s=50, 
               color=PROFESSIONAL_COLORS['secondary'], edgecolors='white', linewidth=0.5, zorder=2)
    # Add trend line
    z = np.polyfit(merged['volume'], merged['mean_hippo'], 1)
    p = np.poly1d(z)
    ax0.plot(merged['volume'], p(merged['volume']), color=PROFESSIONAL_COLORS['accent'], 
            alpha=0.8, linewidth=2, linestyle='--', zorder=1)
    ax0.set_xlabel('Bone Volume', fontsize=8, color=PROFESSIONAL_COLORS['primary'])
    ax0.set_ylabel('Hippocampus EF (V/m)', fontsize=8, color=PROFESSIONAL_COLORS['primary'])
    ax0.set_title(f'A. Hippocampus EF vs Bone Volume\nr={hippo_volume_corr:.3f}, p={hippo_volume_p:.3f}', 
                  fontsize=9, fontweight='bold', color=PROFESSIONAL_COLORS['primary'])
    ax0.tick_params(axis='both', labelsize=7, color=PROFESSIONAL_COLORS['primary'])
    ax0.grid(False)

    # Panel B: Hippocampus vs Bone Thickness
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(merged['mean'], merged['mean_hippo'], alpha=0.7, s=50, 
               color=PROFESSIONAL_COLORS['tertiary'], edgecolors='white', linewidth=0.5, zorder=2)
    # Add trend line
    z = np.polyfit(merged['mean'], merged['mean_hippo'], 1)
    p = np.poly1d(z)
    ax1.plot(merged['mean'], p(merged['mean']), color=PROFESSIONAL_COLORS['accent'], 
            alpha=0.8, linewidth=2, linestyle='--', zorder=1)
    ax1.set_xlabel('Bone Mean Thickness', fontsize=8, color=PROFESSIONAL_COLORS['primary'])
    ax1.set_ylabel('Hippocampus EF (V/m)', fontsize=8, color=PROFESSIONAL_COLORS['primary'])
    ax1.set_title(f'B. Hippocampus EF vs Bone Thickness\nr={hippo_thickness_corr:.3f}, p={hippo_thickness_p:.3f}', 
                  fontsize=9, fontweight='bold', color=PROFESSIONAL_COLORS['primary'])
    ax1.tick_params(axis='both', labelsize=7, color=PROFESSIONAL_COLORS['primary'])
    ax1.grid(False)

    # Panel C: Bone Thickness vs Volume
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(merged['mean'], merged['volume'], alpha=0.7, s=50, 
               color=PROFESSIONAL_COLORS['quaternary'], edgecolors='white', linewidth=0.5, zorder=2)
    # Add trend line
    z = np.polyfit(merged['mean'], merged['volume'], 1)
    p = np.poly1d(z)
    ax2.plot(merged['mean'], p(merged['mean']), color=PROFESSIONAL_COLORS['accent'], 
            alpha=0.8, linewidth=2, linestyle='--', zorder=1)
    ax2.set_xlabel('Bone Mean Thickness', fontsize=8, color=PROFESSIONAL_COLORS['primary'])
    ax2.set_ylabel('Bone Volume', fontsize=8, color=PROFESSIONAL_COLORS['primary'])
    ax2.set_title(f'C. Bone Thickness vs Volume\nr={correlation.iloc[0,1]:.3f}', 
                  fontsize=9, fontweight='bold', color=PROFESSIONAL_COLORS['primary'])
    ax2.tick_params(axis='both', labelsize=7, color=PROFESSIONAL_COLORS['primary'])
    ax2.grid(False)

    # Panel D: Correlation heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    # Create correlation matrix including hippocampus
    corr_matrix = merged[['mean_hippo', 'mean', 'volume']].corr()
    
    # Create heatmap with professional styling
    im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax3.set_xticks(range(len(corr_matrix.columns)))
    ax3.set_yticks(range(len(corr_matrix.columns)))
    ax3.set_xticklabels(['Hippocampus EF', 'Bone Thickness', 'Bone Volume'], 
                        rotation=45, fontsize=7, color=PROFESSIONAL_COLORS['primary'])
    ax3.set_yticklabels(['Hippocampus EF', 'Bone Thickness', 'Bone Volume'],
                        fontsize=7, color=PROFESSIONAL_COLORS['primary'])
    
    # Add correlation values with professional styling
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
            ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha="center", va="center", color=text_color, 
                    fontweight='bold', fontsize=7)
    
    ax3.set_title('D. Correlation Matrix', fontsize=9, fontweight='bold',
                  color=PROFESSIONAL_COLORS['primary'])
    
    # Add colorbar with professional styling
    cbar = plt.colorbar(im, ax=ax3)
    cbar.ax.tick_params(labelsize=7, color=PROFESSIONAL_COLORS['primary'])

    # Panel E: 3D scatter + regression plane
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    x = merged['mean']
    yv = merged['volume']
    z = merged['mean_hippo']
    ax4.scatter(x, yv, z, c=PROFESSIONAL_COLORS['secondary'], marker='o', alpha=0.7, 
               label='Data', s=30, edgecolors='white', linewidth=0.5)
    # Fit regression plane
    X_ = np.column_stack((np.ones(len(x)), x, yv))
    coef_, _, _, _ = np.linalg.lstsq(X_, z, rcond=None)
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 10), np.linspace(yv.min(), yv.max(), 10))
    zz = coef_[0] + coef_[1]*xx + coef_[2]*yy
    ax4.plot_surface(xx, yy, zz, color=PROFESSIONAL_COLORS['accent'], alpha=0.3)
    ax4.set_xlabel('Bone Thickness', fontsize=8, color=PROFESSIONAL_COLORS['primary'])
    ax4.set_ylabel('Bone Volume', fontsize=8, color=PROFESSIONAL_COLORS['primary'])
    ax4.set_zlabel('Hippocampus EF', fontsize=8, color=PROFESSIONAL_COLORS['primary'])
    ax4.set_title('E. 3D Regression', fontsize=9, fontweight='bold',
                  color=PROFESSIONAL_COLORS['primary'])
    ax4.tick_params(axis='both', labelsize=7, color=PROFESSIONAL_COLORS['primary'])

    # Panel F: Observed vs. Predicted
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(merged['mean_hippo'], y_pred, alpha=0.8, s=50, 
               color=PROFESSIONAL_COLORS['secondary'], edgecolors='white', linewidth=0.5, zorder=2)
    min_val = min(merged['mean_hippo'].min(), y_pred.min())
    max_val = max(merged['mean_hippo'].max(), y_pred.max())
    ax5.plot([min_val, max_val], [min_val, max_val], color=PROFESSIONAL_COLORS['accent'], 
            linestyle='--', linewidth=2, label='Ideal fit', zorder=1)
    ax5.set_xlabel('Observed (V/m)', fontsize=8, color=PROFESSIONAL_COLORS['primary'])
    ax5.set_ylabel('Predicted (V/m)', fontsize=8, color=PROFESSIONAL_COLORS['primary'])
    ax5.set_title('F. Observed vs. Predicted', fontsize=9, fontweight='bold',
                  color=PROFESSIONAL_COLORS['primary'])
    ax5.legend(fontsize=7, frameon=False)
    ax5.tick_params(axis='both', labelsize=7, color=PROFESSIONAL_COLORS['primary'])
    ax5.grid(False)

    # Panel G: Text summary (spans bottom row)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    ax6.text(0, 1, summary_text, fontsize=8, va='top', ha='left', wrap=True, 
            family='Arial', color=PROFESSIONAL_COLORS['primary'])
    ax6.set_title('G. Procedure & Results', fontsize=9, fontweight='bold',
                  color=PROFESSIONAL_COLORS['primary'])

    # Add overall title
    fig.suptitle('Hippocampus Electric Field Analysis: Bone Characteristics', 
                fontsize=10, fontweight='bold', color=PROFESSIONAL_COLORS['primary'], y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Publication-ready figure saved as {filename}')

# --- Create figure for hippocampus ---
from pathlib import Path
results_dir = Path(__file__).parent.parent / 'results' / 'figures'
results_dir.mkdir(parents=True, exist_ok=True)
create_publication_figure(results_dir / 'publication_figure_hippo.png')

# --- Print summary statistics ---
print("\n=== CORRELATION SUMMARY ===")
print(f"Hippocampus EF vs Bone Volume: r={hippo_volume_corr:.3f}, p={hippo_volume_p:.3f}")
print(f"Hippocampus EF vs Bone Thickness: r={hippo_thickness_corr:.3f}, p={hippo_thickness_p:.3f}")
print(f"Bone Thickness vs Volume: r={correlation.iloc[0,1]:.3f}")
print(f"Sample size: n={len(merged)}") 
