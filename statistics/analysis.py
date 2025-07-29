
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
print("Loading data...")

# Define input directory - CHANGE THIS LINE TO SWITCH EXPERIMENTS
input_dir = './experiment_insula/'  # Options: './experiment_1/' or './experiment_2/'
csv_a_path = input_dir + 'condition_opt.csv'
csv_b_path = input_dir + 'condition_mapped.csv'

df_a = pd.read_csv(csv_a_path)
df_b = pd.read_csv(csv_b_path)

print(f"Loading data from: {input_dir}")

# Remove AVERAGE row and extract focality from the 4th column
df_a = df_a[df_a['Subject_ID'] != 'AVERAGE'].copy()
df_a['focality'] = df_a.iloc[:, 3]  # Get focality from 4th column
df_a = df_a[['Subject_ID', 'mean', 'max', 'focality']].copy()

df_b = df_b[df_b['Subject_ID'] != 'AVERAGE'][['Subject_ID', 'mean', 'max', 'focality']].copy()

# Add condition labels and combine
df_a['condition'] = 'A'
df_b['condition'] = 'B'
df = pd.concat([df_a, df_b], ignore_index=True)

print(f"Analyzing {len(df_a)} participants across 2 conditions")

# Statistical analysis
variables = ['mean', 'max', 'focality']
results = {}
text_caption = f"EXPERIMENTAL RESULTS CAPTION\n"
text_caption += f"=" * 50 + "\n\n"
text_caption += f"Study Design: Within-subjects comparison (N = {len(df_a)} participants)\n"
text_caption += f"Variables Analyzed: {', '.join(variables)}\n"
text_caption += f"Conditions: A vs B\n\n"

print("\n=== RESULTS ===")
print("Format: Variable: direction (raw_diff = %_change = z_score), p-value - significance")
print("-" * 80)
for var in variables:
    # Get data
    a_data = df[df['condition'] == 'A'][var].values
    b_data = df[df['condition'] == 'B'][var].values
    
    # Basic stats
    mean_a, mean_b = np.mean(a_data), np.mean(b_data)
    std_a, std_b = np.std(a_data, ddof=1), np.std(b_data, ddof=1)
    diff = mean_b - mean_a
    
    # Calculate percentage change and z-score
    percent_change = (diff / mean_a) * 100
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)  # Pooled standard deviation
    z_score = diff / pooled_std
    
    # Statistical test
    t_stat, p_value = stats.ttest_rel(b_data, a_data)
    cohens_d = np.mean(b_data - a_data) / np.std(b_data - a_data, ddof=1)
    
    # Simple interpretation
    sig = "**Significant**" if p_value < 0.05 else "Not significant"
    direction = "higher" if diff > 0 else "lower"
    
    print(f"{var.upper()}: {direction} in B ({diff:.3f} = {percent_change:+.1f}% = {z_score:+.2f} SD), p={p_value:.3f} - {sig}")
    
    # Store results
    results[var] = {'diff': diff, 'percent_change': percent_change, 'z_score': z_score, 
                    'p': p_value, 'cohens_d': cohens_d, 'sig': p_value < 0.05}
    
    # Add to text caption
    text_caption += f"{var.upper()}: "
    text_caption += f"Condition B shows {direction} {var} than Condition A ({percent_change:+.1f}% change, "
    text_caption += f"p = {p_value:.3f}, Cohen's d = {cohens_d:.2f}). {sig.replace('*', '').strip()}\n"

# Create publication-ready individual variable plots
plt.style.use('default')  # Clean publication style
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, var in enumerate(variables):
    ax = axes[i]
    
    # Get paired data for each participant
    pivot_df = df.pivot(index='Subject_ID', columns='condition', values=var).dropna()
    
    # Calculate means and standard errors
    mean_a = pivot_df['A'].mean()
    mean_b = pivot_df['B'].mean()
    sem_a = pivot_df['A'].sem()
    sem_b = pivot_df['B'].sem()
    
    # Create bar plot for means
    x_pos = [0, 1]
    means = [mean_a, mean_b]
    sems = [sem_a, sem_b]
    
    # Color bars based on significance
    bar_color = '#2E8B57' if results[var]['sig'] else '#708090'  # Sea green if significant, slate gray if not
    
    bars = ax.bar(x_pos, means, yerr=sems, capsize=8, color=bar_color, alpha=0.8, 
                  width=0.6, edgecolor='black', linewidth=1.2)
    
    # Add individual participant trajectories
    for idx, (_, row) in enumerate(pivot_df.iterrows()):
        # Add slight jitter to x-coordinates for visibility
        jitter = np.random.normal(0, 0.02, 2)
        x_jittered = [x_pos[0] + jitter[0], x_pos[1] + jitter[1]]
        
        # Plot line connecting conditions for each participant
        ax.plot(x_jittered, [row['A'], row['B']], 'o-', 
                color='gray', alpha=0.4, linewidth=1, markersize=4)
    
    # Customize axes
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Condition A', 'Condition B'], fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{var.capitalize()}', fontsize=12, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Statistical annotation
    p_val = results[var]['p']
    percent_change = results[var]['percent_change']
    
    # Determine significance level for annotation
    if p_val < 0.001:
        sig_text = '***'
    elif p_val < 0.01:
        sig_text = '**'
    elif p_val < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'
    
    # Add significance annotation above bars
    y_max = max(means) + max(sems)
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_text = y_max + y_range * 0.1
    
    # Statistical bracket
    ax.plot([0, 1], [y_text, y_text], 'k-', linewidth=1.5)
    ax.plot([0, 0], [y_text - y_range*0.02, y_text], 'k-', linewidth=1.5)
    ax.plot([1, 1], [y_text - y_range*0.02, y_text], 'k-', linewidth=1.5)
    
    # Add statistical text with improved spacing
    ax.text(0.5, y_text + y_range * 0.03, sig_text, ha='center', va='bottom', 
            fontsize=14, fontweight='bold')
    ax.text(0.5, y_text + y_range * 0.10, f'p = {p_val:.3f}', ha='center', va='bottom', 
            fontsize=10)
    ax.text(0.5, y_text + y_range * 0.16, f'{percent_change:+.1f}%', ha='center', va='bottom', 
            fontsize=10, fontweight='bold', color=bar_color)
    
    # Set title with effect size information
    cohen_d = results[var]['cohens_d']
    ax.set_title(f'{var.capitalize()}\n(Cohen\'s d = {cohen_d:.2f})', 
                fontsize=13, fontweight='bold', pad=60)
    
    # Improve y-axis formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

# Overall figure formatting
plt.suptitle('Experimental Results: Condition A vs Condition B', 
             fontsize=16, fontweight='bold', y=1.02)

# Add figure legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='gray', alpha=0.6, linewidth=2, 
           label='Individual participants'),
    plt.Rectangle((0,0),1,1, facecolor='#2E8B57', alpha=0.8, 
                  label='Significant difference'),
    plt.Rectangle((0,0),1,1, facecolor='#708090', alpha=0.8, 
                  label='Non-significant difference')
]

fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.05), 
           ncol=3, fontsize=11, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make room for legend

# Save with high quality for publication in the same directory as CSV files
plt.savefig(input_dir + 'publication_results.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig(input_dir + 'publication_results.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print(f"\nPublication-ready plots saved as:")
print(f"  - {input_dir}publication_results.png (high-res)")
print(f"  - {input_dir}publication_results.pdf (vector format)")



# Create simple summary
text_caption += f"\nSUMMARY:\n"
sig_vars = [var for var, res in results.items() if res['sig']]
if sig_vars:
    text_caption += f"Significant differences found in: {', '.join(sig_vars)}\n"
else:
    text_caption += f"No significant differences found between conditions.\n"

text_caption += f"\nSTATISTICAL ANALYSIS:\n"
text_caption += f"Paired t-tests were performed to compare conditions A and B for each variable, "
text_caption += f"accounting for the within-subjects design. Effect sizes were calculated using "
text_caption += f"Cohen's d for paired samples.\n"

text_caption += f"\nFIGURE DESCRIPTION:\n"
text_caption += f"Bar plots show group means ± standard error for each variable. "
text_caption += f"Gray lines connect individual participant responses between conditions. "
text_caption += f"Green bars indicate statistically significant differences (p < 0.05), "
text_caption += f"gray bars indicate non-significant differences. Statistical brackets show "
text_caption += f"significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001.\n"

# Save text caption
with open(input_dir + 'results_caption.txt', 'w') as f:
    f.write(text_caption)

print(f"Results caption saved as: {input_dir}results_caption.txt")

# Print final summary
print("\n=== STANDARDIZED EFFECT SUMMARY ===")
for var in variables:
    percent = results[var]['percent_change']
    z_score = results[var]['z_score']
    sig_marker = "***" if results[var]['p'] < 0.001 else "**" if results[var]['p'] < 0.01 else "*" if results[var]['sig'] else ""
    
    print(f"{var.capitalize():10}: {percent:+5.1f}% ({z_score:+.2f} SD) {sig_marker}")

print("\nLegend: * p<0.05, ** p<0.01, *** p<0.001")
print("Analysis complete!")

plt.show()
