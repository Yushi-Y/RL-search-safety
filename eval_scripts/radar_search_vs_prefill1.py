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

def create_radar_chart_search_vs_prefill1():
    """Create a comprehensive radar chart with 4 subplots comparing Search vs Prefill 1"""
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
    
    # Define metrics and angles
    metrics = ['Refusal', 'Answer safety', 'Search safety']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Define colors for each condition (Search = Green, Prefill 1 = Red)
    colors = {
        'Search': '#2ca02c',  # Green
        'Prefill 1': '#d62728'  # Red
    }
    
    # Define subplot positions and titles
    subplot_configs = [
        (0, 0, 'Qwen-2.5-7B-IT + Local Search'),
        (0, 1, 'Llama-3.2-3B-IT + Local Search'),
        (1, 0, 'Qwen-2.5-7B-IT + Web Search'),
        (1, 1, 'Llama-3.2-3B-IT + Web Search')
    ]
    
    for i, (row, col, title) in enumerate(subplot_configs):
        ax = axes[row, col]
        model_key = list(data.keys())[i]
        model_data = data[model_key]
        
        # Plot each condition
        for condition, color in colors.items():
            if condition in model_data:
                values = []
                for metric in metrics:
                    value = model_data[condition][metric]
                    if not np.isnan(value):
                        values.append(value)
                    else:
                        values.append(0)  # Use 0 for missing values
                
                values += values[:1]  # Complete the circle
                
                # Plot the line with different styles to show size difference (no filling)
                if condition == 'Search':
                    # Make Search larger and more prominent
                    ax.plot(angles, values, 'o-', linewidth=3, label=condition, 
                           color=color, markersize=7, alpha=0.9)
                else:
                    # Make other conditions smaller
                    ax.plot(angles, values, 'o-', linewidth=2, label=condition, 
                           color=color, markersize=5, alpha=0.7)
        
        # Customize the subplot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([''] * len(metrics))  # Remove all metric labels
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Customize grid with larger numbers and add 0
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_yticklabels(['0', '20', '40', '60', '80', '100'], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on the outer ring
        ax.set_theta_offset(np.pi / 2)  # Start from top
        ax.set_theta_direction(-1)  # Clockwise
    
    # Add a single legend for the entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    fig.legend(unique_handles, unique_labels, loc='upper right', bbox_to_anchor=(1.05, 0.95), fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/data/kebl6672/AGENTIC-RL/eval_scripts/radar_search_vs_prefill1.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Radar chart (Search vs Prefill 1) saved to: /data/kebl6672/AGENTIC-RL/eval_scripts/radar_search_vs_prefill1.png")

if __name__ == "__main__":
    create_radar_chart_search_vs_prefill1()