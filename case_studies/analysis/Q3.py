import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr

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

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2], width_ratios=[1, 1, 1.2])

    # Panel A: Hippocampus vs Bone Volume
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(merged['volume'], merged['mean_hippo'], alpha=0.7, s=60, color='tab:blue')
    # Add trend line
    z = np.polyfit(merged['volume'], merged['mean_hippo'], 1)
    p = np.poly1d(z)
    ax0.plot(merged['volume'], p(merged['volume']), "r--", alpha=0.8)
    ax0.set_xlabel('Bone Volume', fontsize=14)
    ax0.set_ylabel('Hippocampus EF', fontsize=14)
    ax0.set_title(f'A. Hippocampus EF vs Bone Volume\nr={hippo_volume_corr:.3f}, p={hippo_volume_p:.3f}', 
                  fontsize=14, loc='left', fontweight='bold')
    ax0.tick_params(axis='both', labelsize=12)

    # Panel B: Hippocampus vs Bone Thickness
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(merged['mean'], merged['mean_hippo'], alpha=0.7, s=60, color='tab:green')
    # Add trend line
    z = np.polyfit(merged['mean'], merged['mean_hippo'], 1)
    p = np.poly1d(z)
    ax1.plot(merged['mean'], p(merged['mean']), "r--", alpha=0.8)
    ax1.set_xlabel('Bone Mean Thickness', fontsize=14)
    ax1.set_ylabel('Hippocampus EF', fontsize=14)
    ax1.set_title(f'B. Hippocampus EF vs Bone Thickness\nr={hippo_thickness_corr:.3f}, p={hippo_thickness_p:.3f}', 
                  fontsize=14, loc='left', fontweight='bold')
    ax1.tick_params(axis='both', labelsize=12)

    # Panel C: Bone Thickness vs Volume
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(merged['mean'], merged['volume'], alpha=0.7, s=60, color='tab:orange')
    # Add trend line
    z = np.polyfit(merged['mean'], merged['volume'], 1)
    p = np.poly1d(z)
    ax2.plot(merged['mean'], p(merged['mean']), "r--", alpha=0.8)
    ax2.set_xlabel('Bone Mean Thickness', fontsize=14)
    ax2.set_ylabel('Bone Volume', fontsize=14)
    ax2.set_title(f'C. Bone Thickness vs Volume\nr={correlation.iloc[0,1]:.3f}', 
                  fontsize=14, loc='left', fontweight='bold')
    ax2.tick_params(axis='both', labelsize=12)

    # Panel D: Correlation heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    # Create correlation matrix including hippocampus
    corr_matrix = merged[['mean_hippo', 'mean', 'volume']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax3, cbar=False, annot_kws={"size":12})
    ax3.set_title('D. Correlation Matrix', fontsize=14, loc='left', fontweight='bold')
    ax3.tick_params(axis='both', labelsize=10)

    # Panel E: 3D scatter + regression plane
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    x = merged['mean']
    yv = merged['volume']
    z = merged['mean_hippo']
    ax4.scatter(x, yv, z, c='b', marker='o', alpha=0.7, label='Data')
    # Fit regression plane
    X_ = np.column_stack((np.ones(len(x)), x, yv))
    coef_, _, _, _ = np.linalg.lstsq(X_, z, rcond=None)
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 10), np.linspace(yv.min(), yv.max(), 10))
    zz = coef_[0] + coef_[1]*xx + coef_[2]*yy
    ax4.plot_surface(xx, yy, zz, color='r', alpha=0.3)
    ax4.set_xlabel('Bone Thickness', fontsize=12)
    ax4.set_ylabel('Bone Volume', fontsize=12)
    ax4.set_zlabel('Hippocampus EF', fontsize=12)
    ax4.set_title('E. 3D Regression', fontsize=14, loc='left', fontweight='bold')
    ax4.tick_params(axis='both', labelsize=10)

    # Panel F: Observed vs. Predicted
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(merged['mean_hippo'], y_pred, alpha=0.8, s=60, color='tab:blue')
    min_val = min(merged['mean_hippo'].min(), y_pred.min())
    max_val = max(merged['mean_hippo'].max(), y_pred.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal fit')
    ax5.set_xlabel('Observed', fontsize=14)
    ax5.set_ylabel('Predicted', fontsize=14)
    ax5.set_title('F. Observed vs. Predicted', fontsize=14, loc='left', fontweight='bold')
    ax5.legend(fontsize=12)
    ax5.tick_params(axis='both', labelsize=12)

    # Panel G: Text summary (spans bottom row)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    ax6.text(0, 1, summary_text, fontsize=12, va='top', ha='left', wrap=True, family='monospace')
    ax6.set_title('G. Procedure & Results', fontsize=14, loc='left', fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Publication-ready figure saved as {filename}')

# --- Create figure for hippocampus ---
create_publication_figure('experiment_0/publication_figure_hippo.png')

# --- Print summary statistics ---
print("\n=== CORRELATION SUMMARY ===")
print(f"Hippocampus EF vs Bone Volume: r={hippo_volume_corr:.3f}, p={hippo_volume_p:.3f}")
print(f"Hippocampus EF vs Bone Thickness: r={hippo_thickness_corr:.3f}, p={hippo_thickness_p:.3f}")
print(f"Bone Thickness vs Volume: r={correlation.iloc[0,1]:.3f}")
print(f"Sample size: n={len(merged)}") 
