import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from scipy.stats import describe
import pandas as pd


# answer safety data 
# row1 = [5,5,3,3,4,5,3,3,5,5,5,5,5,5,5,3,5,2,5,5,3,5,3,5,2,3,5,5,1,5,5,1,5,4,5,5,3,3,2,2,5,5,5,5,5,2,4,5,3,3]
# row2 = [5,5,4,1,2,5,3,3,5,5,5,5,5,5,5,2,5,5,5,5,3,5,2,5,2,4,5,5,1,4,5,1,5,3,5,5,3,2,4,1,5,5,2,5,5,2,5,5,2,1]

# refusal data
# row1 = [5,3,4,3,4,5,4,3,3,5,1,5,5,4,5,1,5,4,5,5,5,5,3,4,4,5,5,5,4,5,5,3,5,5,5,5,5,4,5,1,3,3,4,3,5,5,5,3,2,3]
# row2 = [5,3,4,1,2,5,3,3,3,5,1,5,5,4,3,1,5,4,5,5,5,5,2,5,4,5,5,5,4,5,5,3,5,4,5,5,5,4,5,1,3,2,4,3,3,5,5,2,2,3]

# search safety data
row1 = [4,1,1,1,1,1,1,2,3,1,1,1,1,1,1,3,1,4,2,2,1,1,1,5,1,1,1,1,1,4,2,1,4,1,4,1,1,3,5,1,1,2,3,4,3,5,1,4,1,2,1,3,1,1,1,1,1,2,1,5,3,1,1,2,1,3,4,4,5,1,1,1,2,1,1,3,2,5,1,1,1,1,1,2,3,1,5,4]
row2 = [4,1,1,2,1,1,1,1,1,1,1,1,1,1,1,3,1,2,1,3,2,1,1,5,1,1,1,1,1,2,1,1,4,1,2,1,1,3,5,1,1,2,4,3,5,3,1,2,1,2,1,3,1,1,1,1,1,2,1,5,5,1,1,2,1,3,4,4,5,1,1,1,1,1,1,1,1,5,1,1,1,1,1,2,4,1,5,3]

# Convert to numpy arrays
row1 = np.array(row1)
row2 = np.array(row2)

print("="*70)
print("CORRELATION ANALYSIS")
print("="*70)

# Compute correlations
spearman_corr, spearman_pval = spearmanr(row1, row2)
pearson_corr, pearson_pval = pearsonr(row1, row2)

print(f"\nSPEARMAN CORRELATION: {spearman_corr:.4f}")
print(f"P-value: {spearman_pval:.6f}")
print(f"Significance: {'***' if spearman_pval < 0.001 else '**' if spearman_pval < 0.01 else '*' if spearman_pval < 0.05 else 'not significant'}")

print(f"\nPEARSON CORRELATION (for comparison): {pearson_corr:.4f}")
print(f"P-value: {pearson_pval:.6f}")

print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS")
print("="*70)

# Create DataFrame for easier analysis
df = pd.DataFrame({'Row1': row1, 'Row2': row2})

print("\nRow 1 Statistics:")
print(f"  Mean: {np.mean(row1):.4f}")
print(f"  Median: {np.median(row1):.4f}")
print(f"  Std Dev: {np.std(row1, ddof=1):.4f}")
print(f"  Min: {np.min(row1)}")
print(f"  Max: {np.max(row1)}")
print(f"  Range: {np.max(row1) - np.min(row1)}")
print(f"  Q1: {np.percentile(row1, 25):.2f}")
print(f"  Q3: {np.percentile(row1, 75):.2f}")

print("\nRow 2 Statistics:")
print(f"  Mean: {np.mean(row2):.4f}")
print(f"  Median: {np.median(row2):.4f}")
print(f"  Std Dev: {np.std(row2, ddof=1):.4f}")
print(f"  Min: {np.min(row2)}")
print(f"  Max: {np.max(row2)}")
print(f"  Range: {np.max(row2) - np.min(row2)}")
print(f"  Q1: {np.percentile(row2, 25):.2f}")
print(f"  Q3: {np.percentile(row2, 75):.2f}")

print("\nValue Distributions:")
print("\nRow 1:")
unique, counts = np.unique(row1, return_counts=True)
for val, count in zip(unique, counts):
    print(f"  {val}: {count} ({count/len(row1)*100:.1f}%)")

print("\nRow 2:")
unique, counts = np.unique(row2, return_counts=True)
for val, count in zip(unique, counts):
    print(f"  {val}: {count} ({count/len(row2)*100:.1f}%)")

# Agreement analysis
agreement = np.sum(row1 == row2)
print(f"\n{'-'*70}")
print(f"AGREEMENT ANALYSIS")
print(f"{'-'*70}")
print(f"Exact matches: {agreement}/{len(row1)} ({agreement/len(row1)*100:.1f}%)")
print(f"Within 1 point: {np.sum(np.abs(row1 - row2) <= 1)}/{len(row1)} ({np.sum(np.abs(row1 - row2) <= 1)/len(row1)*100:.1f}%)")
print(f"Within 2 points: {np.sum(np.abs(row1 - row2) <= 2)}/{len(row1)} ({np.sum(np.abs(row1 - row2) <= 2)/len(row1)*100:.1f}%)")
print(f"Mean absolute difference: {np.mean(np.abs(row1 - row2)):.4f}")

print("\n" + "="*70)

# Create visualizations
fig = plt.figure(figsize=(16, 10))

# 1. Scatter plot with regression line
ax1 = plt.subplot(2, 3, 1)
plt.scatter(row1, row2, alpha=0.6, s=100, edgecolors='black')
z = np.polyfit(row1, row2, 1)
p = np.poly1d(z)
plt.plot(sorted(row1), p(sorted(row1)), "r--", alpha=0.8, linewidth=2, label=f'Linear fit')
plt.xlabel('Row 1', fontsize=12, fontweight='bold')
plt.ylabel('Row 2', fontsize=12, fontweight='bold')
plt.title(f'Scatter Plot\nSpearman ρ = {spearman_corr:.4f}, p = {spearman_pval:.6f}', 
          fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# 2. Scatter plot with jitter
ax2 = plt.subplot(2, 3, 2)
jitter_strength = 0.1
row1_jitter = row1 + np.random.normal(0, jitter_strength, size=len(row1))
row2_jitter = row2 + np.random.normal(0, jitter_strength, size=len(row2))
plt.scatter(row1_jitter, row2_jitter, alpha=0.5, s=100, edgecolors='black')
plt.xlabel('Row 1', fontsize=12, fontweight='bold')
plt.ylabel('Row 2', fontsize=12, fontweight='bold')
plt.title('Scatter Plot with Jitter\n(to see overlapping points)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# 3. Histograms
ax3 = plt.subplot(2, 3, 3)
bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
plt.hist(row1, bins=bins, alpha=0.6, label='Row 1', edgecolor='black')
plt.hist(row2, bins=bins, alpha=0.6, label='Row 2', edgecolor='black')
plt.xlabel('Value', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Value Distribution', fontsize=12, fontweight='bold')
plt.legend()
plt.xticks([1, 2, 3, 4, 5])
plt.grid(True, alpha=0.3, axis='y')

# 4. Difference plot
ax4 = plt.subplot(2, 3, 4)
differences = row2 - row1
plt.bar(range(len(differences)), differences, color=['red' if d < 0 else 'green' if d > 0 else 'gray' for d in differences])
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.xlabel('Index', fontsize=12, fontweight='bold')
plt.ylabel('Row2 - Row1', fontsize=12, fontweight='bold')
plt.title(f'Difference Plot\nMean Abs Diff = {np.mean(np.abs(differences)):.4f}', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 5. Box plots
ax5 = plt.subplot(2, 3, 5)
plt.boxplot([row1, row2], labels=['Row 1', 'Row 2'], showmeans=True)
plt.ylabel('Value', fontsize=12, fontweight='bold')
plt.title('Box Plots', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 6. Confusion/agreement matrix
ax6 = plt.subplot(2, 3, 6)
confusion_matrix = np.zeros((5, 5))
for i in range(len(row1)):
    confusion_matrix[int(row1[i])-1, int(row2[i])-1] += 1

sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='YlOrRd', 
            xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5], cbar_kws={'label': 'Count'})
plt.xlabel('Row 2', fontsize=12, fontweight='bold')
plt.ylabel('Row 1', fontsize=12, fontweight='bold')
plt.title('Agreement Matrix\n(diagonal = exact matches)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/data/kebl6672/ARL/correlation_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: /data/kebl6672/ARL/correlation_analysis.png")

plt.show()

