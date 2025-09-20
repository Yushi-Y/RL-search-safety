import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Data from the provided tables (already in 0-100 scale)
data = {
    'Qwen-Local': {
        'Search': {'Refusal': 92.0, 'Answer safety': 87.8, 'Search safety': 70.8},
        'Prompt 1': {'Refusal': 71.5, 'Answer safety': 65.0, 'Search safety': 31.5},
        'Prefill 1': {'Refusal': 76.8, 'Answer safety': 73.3, 'Search safety': 33.3},
        'Prefill 1 + Ten': {'Refusal': 61.8, 'Answer safety': 50.7, 'Search safety': 34.8}
    },
    'Llama-Local': {
        'Search': {'Refusal': 97.0, 'Answer safety': 96.3, 'Search safety': 37.0},
        'Prompt 1': {'Refusal': 79.3, 'Answer safety': 80.5, 'Search safety': 17.3},
        'Prefill 1': {'Refusal': 74.0, 'Answer safety': 73.3, 'Search safety': 14.0},
        'Prefill 1 + Ten': {'Refusal': 67.0, 'Answer safety': 64.3, 'Search safety': 46.5}
    },
    'Qwen-Web': {
        'Search': {'Refusal': 91.0, 'Answer safety': 91.0, 'Search safety': 64.2},
        'Prompt 1': {'Refusal': 74.0, 'Answer safety': 73.8, 'Search safety': 35.0},
        'Prefill 1': {'Refusal': 78.0, 'Answer safety': 74.8, 'Search safety': 30.0},
        'Prefill 1 + Ten': {'Refusal': 62.0, 'Answer safety': 55.2, 'Search safety': 35.0}
    },
    'Llama-Web': {
        'Search': {'Refusal': 97.2, 'Answer safety': 96.2, 'Search safety': 35.8},
        'Prompt 1': {'Refusal': 84.7, 'Answer safety': 85.8, 'Search safety': 26.7},
        'Prefill 1': {'Refusal': 75.5, 'Answer safety': 75.0, 'Search safety': 13.5},
        'Prefill 1 + Ten': {'Refusal': 64.2, 'Answer safety': 60.5, 'Search safety': 35.0}
    }
}

def create_dumbbell_search_min_prefill1_ten():
    """Create a 2x2 grid of dumbbell plots comparing Search, min(Prompt 1, Prefill 1), and Prefill 1 + Ten"""
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Safety Metrics Comparison: Search vs Min(Prompt 1, Prefill 1) vs Prefill 1 + Ten', fontsize=16, fontweight='bold', y=0.95)
    
    # Define metrics and colors (removed Search safety)
    metrics = ['Refusal', 'Answer safety']
    colors = {
        'Search': '#ff7f0e',  # Orange - highest
        'Min(Prompt 1, Prefill 1)': '#2ca02c',  # Green - middle
        'Prefill 1 + Ten': '#d62728'  # Red - lowest
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
        prompt1_values = [model_data['Prompt 1'][metric] for metric in metrics]
        prefill1_values = [model_data['Prefill 1'][metric] for metric in metrics]
        prefill1_ten_values = [model_data['Prefill 1 + Ten'][metric] for metric in metrics]
        
        # Calculate min(Prompt 1, Prefill 1) for each metric
        min_values = [min(prompt1_values[j], prefill1_values[j]) for j in range(len(metrics))]
        
        # Set up x positions
        x = np.arange(len(metrics))
        
        # Plot connecting lines (dumbbell stems) - connect all three points
        for j in range(len(metrics)):
            # Connect Search to Min
            ax.plot([j, j], [min_values[j], search_values[j]], 
                   color='gray', linewidth=2, alpha=0.7)
            # Connect Min to Prefill 1 + Ten
            ax.plot([j, j], [prefill1_ten_values[j], min_values[j]], 
                   color='lightgray', linewidth=1.5, alpha=0.5)
        
        # Plot Search points (orange) - highest values
        search_points = ax.scatter(x, search_values, s=150, c=colors['Search'], 
                                 label='Search', alpha=0.9, edgecolor='black', linewidth=2)
        
        # Plot Min points (green) - middle values
        min_points = ax.scatter(x, min_values, s=120, c=colors['Min(Prompt 1, Prefill 1)'], 
                               label='Min(Prompt 1, Prefill 1)', alpha=0.9, edgecolor='black', linewidth=1.5)
        
        # Plot Prefill 1 + Ten points (red) - lowest values
        prefill1_ten_points = ax.scatter(x, prefill1_ten_values, s=120, c=colors['Prefill 1 + Ten'], 
                                        label='Prefill 1 + Ten', alpha=0.9, edgecolor='black', linewidth=1.5)
        
        # Add numeric labels at each point
        for j, (search_val, min_val, prefill1_ten_val) in enumerate(zip(search_values, min_values, prefill1_ten_values)):
            # Search labels (orange) - above
            ax.text(j, search_val + 3, f'{search_val:.1f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
            # Min labels (green) - at the point
            ax.text(j, min_val, f'{min_val:.1f}', 
                   ha='center', va='center', fontweight='bold', fontsize=9, color='white')
            # Prefill 1 + Ten labels (red) - below
            ax.text(j, prefill1_ten_val - 3, f'{prefill1_ten_val:.1f}', 
                   ha='center', va='top', fontweight='bold', fontsize=9)
        
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
            ax.legend(loc='upper right', fontsize=10)
    
    # Add summary annotation
    fig.text(0.98, 0.02, 'Search (orange) > Min(Prompt 1, Prefill 1) (green) > Prefill 1 + Ten (red) across Refusal and Answer Safety metrics.', 
             ha='right', va='bottom', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # Add arrow pointing to a representative gap
    axes[0, 1].annotate('', xy=(0.75, 0.15), xytext=(0.85, 0.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                transform=fig.transFigure)
    
    plt.tight_layout()
    plt.savefig('/data/kebl6672/AGENTIC-RL/eval_scripts/dumbbell_search_min_prefill1_ten.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Dumbbell plots (Search vs Min vs Prefill 1 + Ten) saved to: /data/kebl6672/AGENTIC-RL/eval_scripts/dumbbell_search_min_prefill1_ten.png")

if __name__ == "__main__":
    create_dumbbell_search_min_prefill1_ten()