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
        'IT': {'Refusal': 95.8, 'Answer safety': 95.0, 'Search safety': np.nan},
        'Search': {'Refusal': 92.0, 'Answer safety': 87.8, 'Search safety': 70.8},
        'Prompt (avg)': {'Refusal': 74.2, 'Answer safety': 70.7, 'Search safety': 36.0},
        'Prefill (avg)': {'Refusal': 74.3, 'Answer safety': 69.4, 'Search safety': 28.6},
        'Prefill+Ten (avg)': {'Refusal': 62.4, 'Answer safety': 51.8, 'Search safety': 34.3}
    },
    'Llama-Local': {
        'IT': {'Refusal': 98.8, 'Answer safety': 97.2, 'Search safety': np.nan},
        'Search': {'Refusal': 97.0, 'Answer safety': 96.3, 'Search safety': 37.0},
        'Prompt (avg)': {'Refusal': 84.3, 'Answer safety': 84.5, 'Search safety': 21.8},
        'Prefill (avg)': {'Refusal': 76.7, 'Answer safety': 72.2, 'Search safety': 13.8},
        'Prefill+Ten (avg)': {'Refusal': 71.7, 'Answer safety': 63.9, 'Search safety': 48.8}
    },
    'Qwen-Web': {
        'IT': {'Refusal': 95.8, 'Answer safety': 95.0, 'Search safety': np.nan},
        'Search': {'Refusal': 91.0, 'Answer safety': 91.0, 'Search safety': 64.2},
        'Prompt (avg)': {'Refusal': 77.5, 'Answer safety': 76.8, 'Search safety': 34.4},
        'Prefill (avg)': {'Refusal': 74.6, 'Answer safety': 68.4, 'Search safety': 25.8},
        'Prefill+Ten (avg)': {'Refusal': 66.3, 'Answer safety': 53.5, 'Search safety': 34.6}
    },
    'Llama-Web': {
        'IT': {'Refusal': 98.8, 'Answer safety': 97.0, 'Search safety': np.nan},
        'Search': {'Refusal': 97.2, 'Answer safety': 96.2, 'Search safety': 35.8},
        'Prompt (avg)': {'Refusal': 85.8, 'Answer safety': 86.7, 'Search safety': 28.8},
        'Prefill (avg)': {'Refusal': 77.2, 'Answer safety': 71.8, 'Search safety': 14.3},
        'Prefill+Ten (avg)': {'Refusal': 69.9, 'Answer safety': 59.7, 'Search safety': 29.4}
    }
}

def create_comprehensive_radar_chart():
    """Create a comprehensive radar chart with 4 subplots for each model+search combination"""
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
    
    # Define metrics and angles
    metrics = ['Refusal', 'Answer safety', 'Search safety']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Define colors for each condition (removed IT and Prefill+Ten)
    colors = {
        'Search': '#ff7f0e',  # Orange
        'Prompt (avg)': '#2ca02c',  # Green
        'Prefill (avg)': '#d62728'  # Red
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
                
                # Plot the line with different styles to show size difference
                if condition == 'Search':
                    # Make Search larger and more prominent
                    ax.plot(angles, values, 'o-', linewidth=3, label=condition, 
                           color=color, markersize=7, alpha=0.9)
                    ax.fill(angles, values, alpha=0.25, color=color)
                else:
                    # Make other conditions smaller
                    ax.plot(angles, values, 'o-', linewidth=2, label=condition, 
                           color=color, markersize=5, alpha=0.7)
                    ax.fill(angles, values, alpha=0.15, color=color)
        
        # Customize the subplot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on the outer ring
        ax.set_theta_offset(np.pi / 2)  # Start from top
        ax.set_theta_direction(-1)  # Clockwise
        
        # No individual legends for each subplot
    
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
    plt.savefig('/data/kebl6672/AGENTIC-RL/eval_scripts/comprehensive_radar_chart.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Comprehensive radar chart saved to: /data/kebl6672/AGENTIC-RL/eval_scripts/comprehensive_radar_chart.png")

def create_single_radar_chart():
    """Create a single radar chart with all 4 model+search combinations"""
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Define metrics and angles
    metrics = ['Refusal', 'Answer safety', 'Search safety']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Define colors for each model+search combination
    model_colors = {
        'Qwen-Local': '#1f77b4',  # Blue
        'Llama-Local': '#ff7f0e',  # Orange
        'Qwen-Web': '#2ca02c',  # Green
        'Llama-Web': '#d62728'  # Red
    }
    
    # Define line styles for each condition (removed IT and Prefill+Ten)
    line_styles = {
        'Search': '--',
        'Prompt (avg)': '-.',
        'Prefill (avg)': ':'
    }
    
    # Plot each model+search combination
    for model_key, model_color in model_colors.items():
        model_data = data[model_key]
        
        for condition, line_style in line_styles.items():
            if condition in model_data:
                values = []
                for metric in metrics:
                    value = model_data[condition][metric]
                    if not np.isnan(value):
                        values.append(value)
                    else:
                        values.append(0)
                
                values += values[:1]  # Complete the circle
                
                # Create label
                label = f"{model_key} - {condition}"
                
                # Plot the line with different styles to show size difference
                if condition == 'Search':
                    # Make Search larger and more prominent
                    ax.plot(angles, values, 'o-', linewidth=3, label=label, 
                           color=model_color, markersize=7, alpha=0.9)
                else:
                    # Make other conditions smaller
                    ax.plot(angles, values, 'o-', linewidth=2, label=label, 
                           color=model_color, markersize=5, alpha=0.7)
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.set_theta_offset(np.pi / 2)  # Start from top
    ax.set_theta_direction(-1)  # Clockwise
    
    # Add simplified legend (show all conditions once)
    handles, labels = ax.get_legend_handles_labels()
    unique_condition_labels = {}
    for handle, label in zip(handles, labels):
        # Extract the condition name from labels like "Search (Qwen-Local)"
        condition_name = label.split(' (')[0]
        if condition_name not in unique_condition_labels:
            unique_condition_labels[condition_name] = handle
    
    # Create a new legend with only unique condition names
    ax.legend(unique_condition_labels.values(), unique_condition_labels.keys(),
              bbox_to_anchor=(1.3, 1.0), loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/data/kebl6672/AGENTIC-RL/eval_scripts/single_radar_chart.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Single radar chart saved to: /data/kebl6672/AGENTIC-RL/eval_scripts/single_radar_chart.png")

def create_condition_comparison_radar():
    """Create radar charts comparing each condition across all model+search combinations"""
    
    # Create figure with 3 subplots (one for each condition)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
    
    # Define metrics and angles
    metrics = ['Refusal', 'Answer safety', 'Search safety']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Define colors for each model+search combination
    model_colors = {
        'Qwen-Local': '#1f77b4',  # Blue
        'Llama-Local': '#ff7f0e',  # Orange
        'Qwen-Web': '#2ca02c',  # Green
        'Llama-Web': '#d62728'  # Red
    }
    
    conditions = ['Search', 'Prompt (avg)', 'Prefill (avg)']
    
    # Plot each condition
    for i, condition in enumerate(conditions):
        ax = axes[i]
        
        # Plot each model+search combination for this condition
        for model_key, model_color in model_colors.items():
            model_data = data[model_key]
            
            if condition in model_data:
                values = []
                for metric in metrics:
                    value = model_data[condition][metric]
                    if not np.isnan(value):
                        values.append(value)
                    else:
                        values.append(0)
                
                values += values[:1]  # Complete the circle
                
                # Plot the line
                ax.plot(angles, values, 'o-', linewidth=2.5, label=model_key, 
                       color=model_color, markersize=6, alpha=0.8)
                
                # Fill the area
                ax.fill(angles, values, alpha=0.15, color=model_color)
        
        # Customize the subplot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title(f'{condition}', fontsize=13, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_theta_offset(np.pi / 2)  # Start from top
        ax.set_theta_direction(-1)  # Clockwise
        
        # Add simplified legend (only show model+search combinations, not condition names)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/data/kebl6672/AGENTIC-RL/eval_scripts/condition_comparison_radar.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Condition comparison radar chart saved to: /data/kebl6672/AGENTIC-RL/eval_scripts/condition_comparison_radar.png")

def create_comprehensive_radar_chart_no_prompt():
    """Create a comprehensive radar chart with 4 subplots for each model+search combination (no prompt)"""
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
    
    # Define metrics and angles
    metrics = ['Refusal', 'Answer safety', 'Search safety']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Define colors for each condition (removed IT, Prefill+Ten, and Prompt)
    colors = {
        'Search': '#2ca02c',  # Green
        'Prefill (avg)': '#d62728'  # Red
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
                
                # Plot the line with different styles to show size difference
                if condition == 'Search':
                    # Make Search larger and more prominent
                    ax.plot(angles, values, 'o-', linewidth=3, label=condition, 
                           color=color, markersize=7, alpha=0.9)
                    ax.fill(angles, values, alpha=0.25, color=color)
                else:
                    # Make other conditions smaller
                    ax.plot(angles, values, 'o-', linewidth=2, label=condition, 
                           color=color, markersize=5, alpha=0.7)
                    ax.fill(angles, values, alpha=0.15, color=color)
        
        # Customize the subplot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on the outer ring
        ax.set_theta_offset(np.pi / 2)  # Start from top
        ax.set_theta_direction(-1)  # Clockwise
        
        # No individual legends for each subplot
    
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
    plt.savefig('/data/kebl6672/AGENTIC-RL/eval_scripts/comprehensive_radar_chart_no_prompt.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Comprehensive radar chart (no prompt) saved to: /data/kebl6672/AGENTIC-RL/eval_scripts/comprehensive_radar_chart_no_prompt.png")

def main():
    """Generate all radar chart variations"""
    print("Generating comprehensive radar charts...")
    
    # Create the main 2x2 radar chart
    create_comprehensive_radar_chart()
    
    # Create the main 2x2 radar chart without prompt
    create_comprehensive_radar_chart_no_prompt()
    
    # Create single radar chart with all combinations
    create_single_radar_chart()
    
    # Create condition comparison radar charts
    create_condition_comparison_radar()
    
    print("\nAll radar charts generated successfully!")
    print("Files created:")
    print("- comprehensive_radar_chart.png")
    print("- comprehensive_radar_chart_no_prompt.png")
    print("- single_radar_chart.png")
    print("- condition_comparison_radar.png")

if __name__ == "__main__":
    main()