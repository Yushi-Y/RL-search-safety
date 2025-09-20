import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Updated data from the provided tables (already in 0-100 scale)
data = {
    'Qwen-Local': {
        'Search': {'Refusal': 92.0, 'Answer safety': 87.8, 'Search safety': 70.8},
        'Prefill 1': {'Refusal': 76.8, 'Answer safety': 73.3, 'Search safety': 33.3}
    },
    'Llama-Local': {
        'Search': {'Refusal': 97.0, 'Answer safety': 96.3, 'Search safety': 37.0},
        'Prefill 1': {'Refusal': 74.0, 'Answer safety': 73.3, 'Search safety': 14.0}
    },
    'Qwen-Web': {
        'Search': {'Refusal': 91.0, 'Answer safety': 91.0, 'Search safety': 64.2},
        'Prefill 1': {'Refusal': 78.0, 'Answer safety': 74.8, 'Search safety': 30.0}
    },
    'Llama-Web': {
        'Search': {'Refusal': 97.2, 'Answer safety': 96.2, 'Search safety': 35.8},
        'Prefill 1': {'Refusal': 75.5, 'Answer safety': 75.0, 'Search safety': 13.5}
    }
}

def create_dumbbell_plots():
    """Create a 2x2 grid of dumbbell plots comparing Search vs Prefill 1"""
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Safety Metrics Comparison: Search vs Prefill 1', fontsize=16, fontweight='bold', y=0.95)
    
    # Define metrics and colors
    metrics = ['Refusal', 'Answer safety', 'Search safety']
    colors = {
        'Search': '#ff7f0e',  # Orange - outer/higher
        'Prefill 1': '#d62728'  # Red - inner/lower
    }
    
    # Define subplot positions and titles
    subplot_configs = [
        (0, 0, 'Qwen-2.5-7B-IT + Local Search'),
        (0, 1, 'Llama-3.2-3B-IT + Local Search'),
        (1, 0, 'Qwen-2.5-7B-IT + Web Search'),
        (1, 1, 'Llama-3.2-3B-IT + Web Search')
    ]
    
    # Set consistent y-axis range for all subplots
    y_min, y_max = 0, 100
    
    for i, (row, col, title) in enumerate(subplot_configs):
        ax = axes[row, col]
        model_key = list(data.keys())[i]
        model_data = data[model_key]
        
        # Get data for this model+search combination
        search_values = [model_data['Search'][metric] for metric in metrics]
        prefill_values = [model_data['Prefill 1'][metric] for metric in metrics]
        
        # Set up x positions
        x = np.arange(len(metrics))
        
        # Plot connecting lines (dumbbell stems)
        for j in range(len(metrics)):
            ax.plot([j, j], [prefill_values[j], search_values[j]], 
                   color='gray', linewidth=2, alpha=0.7)
        
        # Plot Search points (orange) - higher values
        search_points = ax.scatter(x, search_values, s=150, c=colors['Search'], 
                                 label='Search', alpha=0.9, edgecolor='black', linewidth=2)
        
        # Plot Prefill 1 points (red) - lower values
        prefill_points = ax.scatter(x, prefill_values, s=120, c=colors['Prefill 1'], 
                                  label='Prefill 1', alpha=0.9, edgecolor='black', linewidth=1.5)
        
        # Add numeric labels at each point
        for j, (search_val, prefill_val) in enumerate(zip(search_values, prefill_values)):
            # Search labels (orange)
            ax.text(j, search_val + 3, f'{search_val:.1f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
            # Prefill 1 labels (red)
            ax.text(j, prefill_val - 5, f'{prefill_val:.1f}', 
                   ha='center', va='top', fontweight='bold', fontsize=10)
        
        # Customize the subplot
        ax.set_xlabel('Safety Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (0-100)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend only to the first subplot
        if i == 0:
            ax.legend(loc='upper right', fontsize=11)
    
    # Add summary annotation
    fig.text(0.98, 0.02, 'Prefill 1 (red) is lower than Search (orange) across all three safety metrics (Refusal, Answer Safety, Search Safety).', 
             ha='right', va='bottom', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # Add arrow pointing to a representative gap
    axes[0, 1].annotate('', xy=(0.75, 0.15), xytext=(0.85, 0.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                transform=fig.transFigure)
    
    plt.tight_layout()
    plt.savefig('/data/kebl6672/AGENTIC-RL/eval_scripts/dumbbell_search_vs_prefill1.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Dumbbell plots (Search vs Prefill 1) saved to: /data/kebl6672/AGENTIC-RL/eval_scripts/dumbbell_search_vs_prefill1.png")

if __name__ == "__main__":
    create_dumbbell_plots()