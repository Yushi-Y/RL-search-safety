import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def convert_scale(value):
    """Convert from 1-5 scale to 0-100 scale"""
    if value == '-':
        return np.nan
    return (float(value) - 1) * 25

# Data from the tables (converted to 0-100 scale)
# Qwen-2.5-7B-IT data
qwen_prompt_data = {
    'Refusal': {
        'IT': convert_scale(4.83),
        'Search': convert_scale(4.68),
        'Prompt 1': convert_scale(3.86),
        'Prompt 1 + Ten searches': convert_scale(3.99),
        'Prompt 2': convert_scale(4.08),
        'Prompt 2 + Ten searches': convert_scale(4.19),
        'Prefill 3': convert_scale(4.70)
    },
    'Answer safety': {
        'IT': convert_scale(4.80),
        'Search': convert_scale(4.51),
        'Prompt 1': convert_scale(3.60),
        'Prompt 1 + Ten searches': convert_scale(3.76),
        'Prompt 2': convert_scale(4.05),
        'Prompt 2 + Ten searches': convert_scale(3.88),
        'Prefill 3': convert_scale(3.70)
    },
    'Search safety': {
        'IT': convert_scale('-'),
        'Search': convert_scale(3.83),
        'Prompt 1': convert_scale(2.26),
        'Prompt 1 + Ten searches': convert_scale(2.17),
        'Prompt 2': convert_scale(2.62),
        'Prompt 2 + Ten searches': convert_scale(2.47),
        'Prefill 3': convert_scale(2.66)
    }
}

qwen_prefill_data = {
    'Refusal': {
        'IT': convert_scale(4.83),
        'Search': convert_scale(4.68),
        'Prefill 1': convert_scale(4.07),
        'Prefill 1 + Ten searches': convert_scale(3.47),
        'Prefill 2': convert_scale(3.87),
        'Prefill 2 + Ten searches': convert_scale(3.52)
    },
    'Answer safety': {
        'IT': convert_scale(4.80),
        'Search': convert_scale(4.51),
        'Prefill 1': convert_scale(3.93),
        'Prefill 1 + Ten searches': convert_scale(3.03),
        'Prefill 2': convert_scale(3.62),
        'Prefill 2 + Ten searches': convert_scale(3.11)
    },
    'Search safety': {
        'IT': convert_scale('-'),
        'Search': convert_scale(3.83),
        'Prefill 1': convert_scale(2.33),
        'Prefill 1 + Ten searches': convert_scale(2.39),
        'Prefill 2': convert_scale(1.95),
        'Prefill 2 + Ten searches': convert_scale(2.35)
    }
}

# Llama-3.2-3B-IT data
llama_prompt_data = {
    'Refusal': {
        'IT': convert_scale(4.95),
        'Search': convert_scale(4.88),
        'Prompt 1': convert_scale(4.17),
        'Prompt 1 + Ten searches': convert_scale(4.27),
        'Prompt 2': convert_scale(4.57),
        'Prompt 2 + Ten searches': convert_scale(4.61),
        'Prefill 3': convert_scale(4.75)
    },
    'Answer safety': {
        'IT': convert_scale(4.89),
        'Search': convert_scale(4.85),
        'Prompt 1': convert_scale(4.22),
        'Prompt 1 + Ten searches': convert_scale(4.36),
        'Prompt 2': convert_scale(4.54),
        'Prompt 2 + Ten searches': convert_scale(4.62),
        'Prefill 3': convert_scale(4.26)
    },
    'Search safety': {
        'IT': convert_scale('-'),
        'Search': convert_scale(2.48),
        'Prompt 1': convert_scale(1.69),
        'Prompt 1 + Ten searches': convert_scale(2.05),
        'Prompt 2': convert_scale(2.05),
        'Prompt 2 + Ten searches': convert_scale(2.60),
        'Prefill 3': convert_scale(1.56)
    }
}

llama_prefill_data = {
    'Refusal': {
        'IT': convert_scale(4.95),
        'Search': convert_scale(4.88),
        'Prefill 1': convert_scale(3.96),
        'Prefill 1 + Ten searches': convert_scale(3.68),
        'Prefill 2': convert_scale(4.17),
        'Prefill 2 + Ten searches': convert_scale(4.05)
    },
    'Answer safety': {
        'IT': convert_scale(4.89),
        'Search': convert_scale(4.85),
        'Prefill 1': convert_scale(3.93),
        'Prefill 1 + Ten searches': convert_scale(3.57),
        'Prefill 2': convert_scale(3.84),
        'Prefill 2 + Ten searches': convert_scale(3.54)
    },
    'Search safety': {
        'IT': convert_scale('-'),
        'Search': convert_scale(2.48),
        'Prefill 1': convert_scale(1.56),
        'Prefill 1 + Ten searches': convert_scale(2.86),
        'Prefill 2': convert_scale(1.54),
        'Prefill 2 + Ten searches': convert_scale(3.04)
    }
}

def create_comparison_figure(model_name, data, save_path):
    """Create comparison figures for a specific model"""
    # Focus on Prefill 1, Prefill 2, Prefill 3 vs IT and Search
    conditions = ['IT', 'Search', 'Prefill 1', 'Prefill 2', 'Prefill 3']
    metrics = ['Refusal', 'Answer safety', 'Search safety']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{model_name} - Prefill Comparison (0-100 Scale)', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract data for this metric
        values = []
        labels = []
        valid_colors = []
        
        for j, condition in enumerate(conditions):
            if condition in data[metric]:
                value = data[metric][condition]
                if not np.isnan(value):
                    values.append(value)
                    labels.append(condition)
                    valid_colors.append(colors[j])
        
        # Create bar plot
        bars = ax.bar(labels, values, color=valid_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Customize plot
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score (0-100)', fontsize=12)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if needed
        if len(labels) > 4:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Figure saved to: {save_path}")

def create_combined_comparison():
    """Create a combined comparison figure for both models"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Prefill Comparison: Qwen-2.5-7B-IT vs Llama-3.2-3B-IT (0-100 Scale)', 
                 fontsize=18, fontweight='bold')
    
    models = [
        ('Qwen-2.5-7B-IT', qwen_prefill_data),
        ('Llama-3.2-3B-IT', llama_prefill_data)
    ]
    
    metrics = ['Refusal', 'Answer safety', 'Search safety']
    conditions = ['IT', 'Search', 'Prefill 1', 'Prefill 2', 'Prefill 3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for model_idx, (model_name, data) in enumerate(models):
        for metric_idx, metric in enumerate(metrics):
            ax = axes[model_idx, metric_idx]
            
            # Extract data for this metric
            values = []
            labels = []
            valid_colors = []
            
            for j, condition in enumerate(conditions):
                if condition in data[metric]:
                    value = data[metric][condition]
                    if not np.isnan(value):
                        values.append(value)
                        labels.append(condition)
                        valid_colors.append(colors[j])
            
            # Create bar plot
            bars = ax.bar(labels, values, color=valid_colors, alpha=0.8, 
                         edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Customize plot
            ax.set_title(f'{model_name} - {metric}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score (0-100)', fontsize=10)
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45, labelsize=9)
    
    plt.tight_layout()
    plt.savefig('/data/kebl6672/AGENTIC-RL/eval_scripts/combined_prefill_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Combined comparison figure saved to: /data/kebl6672/AGENTIC-RL/eval_scripts/combined_prefill_comparison.png")

def create_radar_chart():
    """Create radar charts for both models"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
    fig.suptitle('Prefill Performance Radar Charts (0-100 Scale)', fontsize=16, fontweight='bold')
    
    models = [
        ('Qwen-2.5-7B-IT', qwen_prefill_data),
        ('Llama-3.2-3B-IT', llama_prefill_data)
    ]
    
    metrics = ['Refusal', 'Answer safety', 'Search safety']
    conditions = ['IT', 'Search', 'Prefill 1', 'Prefill 2', 'Prefill 3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for model_idx, (model_name, data) in enumerate(models):
        ax = axes[model_idx]
        
        # Set up angles for radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for condition in conditions:
            values = []
            for metric in metrics:
                if condition in data[metric]:
                    value = data[metric][condition]
                    if not np.isnan(value):
                        values.append(value)
                    else:
                        values.append(0)
                else:
                    values.append(0)
            
            values += values[:1]  # Complete the circle
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=condition, 
                   color=colors[conditions.index(condition) % len(colors)])
            ax.fill(angles, values, alpha=0.25, 
                   color=colors[conditions.index(condition) % len(colors)])
        
        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 100)
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('/data/kebl6672/AGENTIC-RL/eval_scripts/radar_charts.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Radar charts saved to: /data/kebl6672/AGENTIC-RL/eval_scripts/radar_charts.png")

def main():
    """Generate all figures"""
    print("Generating comparison figures...")
    
    # Create individual model figures
    create_comparison_figure('Qwen-2.5-7B-IT', qwen_prefill_data, 
                           '/data/kebl6672/AGENTIC-RL/eval_scripts/qwen_prefill_comparison.png')
    
    create_comparison_figure('Llama-3.2-3B-IT', llama_prefill_data, 
                           '/data/kebl6672/AGENTIC-RL/eval_scripts/llama_prefill_comparison.png')
    
    # Create combined comparison
    create_combined_comparison()
    
    # Create radar charts
    create_radar_chart()
    
    print("\nAll figures generated successfully!")
    print("Files created:")
    print("- qwen_prefill_comparison.png")
    print("- llama_prefill_comparison.png") 
    print("- combined_prefill_comparison.png")
    print("- radar_charts.png")

if __name__ == "__main__":
    main()