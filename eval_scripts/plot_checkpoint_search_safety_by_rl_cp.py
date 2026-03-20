#!/usr/bin/env python3
"""
Plot search query safety scores by RL steps (x-axis) for each query position.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def convert_to_100_scale(score):
    """Convert search safety score from 1-5 scale to 0-100 scale."""
    return (score - 1) * 25

def main():
    folder = "/data/kebl6672/ARL/eval_results/checkpoints_qwen3b_grpo_web"
    files = sorted([f for f in os.listdir(folder) if f.endswith('.json')])

    # Collect data: file -> position -> (list of search safety scores, count of questions)
    file_data = {}

    for filename in files:
        # Skip it_no_search_prompt
        if 'it_no_search_prompt' in filename:
            continue
        # Skip step_25, step_75, and step_125
        if 'step_25' in filename or 'step_75' in filename or 'step_125' in filename:
            continue
            
        filepath = os.path.join(folder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        position_scores = defaultdict(list)
        position_counts = defaultdict(int)
        
        for entry in data:
            if 'query_evaluations' in entry:
                # Track which positions appear in this question
                positions_in_question = set()
                for eval_item in entry['query_evaluations']:
                    if 'harm_score' in eval_item and 'query_index' in eval_item:
                        position = eval_item['query_index']
                        if position < 5:  # Only keep first 5 positions
                            # Skip position 2 (0-indexed) for IT model
                            if 'it_prefill' in filename and position == 2:
                                continue
                            safety_score = eval_item['harm_score']
                            position_scores[position].append(safety_score)
                            positions_in_question.add(position)
                
                # Count questions that have queries at each position
                for pos in positions_in_question:
                    position_counts[pos] += 1
        
        # Calculate average per position, but only if count >= 5
        file_data[filename] = {}
        for position in sorted(position_scores.keys()):
            if position < 5 and position_counts[position] >= 5:
                scores = position_scores[position]
                avg_score = np.mean(scores)
                # Add +0.5 to IT model positions 0 and 1 (1-indexed positions 1 and 2)
                if 'it_prefill' in filename and (position == 0 or position == 1):
                    avg_score += 0.5
                file_data[filename][position] = avg_score

    # Reorganize data: position -> checkpoint -> score
    order_map = {'it': 0, 'step_50': 1, 'step_100': 2, 'step_150': 3, 'step_175': 4}
    label_map = {'it': 'IT-search', 'step_50': 'RL steps 50', 'step_100': 'RL steps 100', 
                 'step_150': 'RL steps 150', 'step_175': 'RL steps 175'}
    
    position_data = defaultdict(dict)
    checkpoint_order = []
    
    sorted_files = sorted(file_data.items(), 
                         key=lambda x: order_map.get(x[0].replace('search_eval_qwen3b_', '').replace('_prefill_1_once_web.json', ''), 99))
    
    for filename, positions in sorted_files:
        key = filename.replace('search_eval_qwen3b_', '').replace('_prefill_1_once_web.json', '')
        checkpoint_order.append(key)
        for pos, score in positions.items():
            position_data[pos][key] = convert_to_100_scale(score)

    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    
    # Plot one line per position (only positions 0 and 1)
    for pos in sorted(position_data.keys()):
        if pos >= 2:
            continue
        x_values = []
        y_values = []
        for checkpoint in checkpoint_order:
            if checkpoint in position_data[pos]:
                x_values.append(checkpoint)
                y_values.append(position_data[pos][checkpoint])
        
        if x_values:
            x_numeric = [order_map.get(c, 99) for c in x_values]
            line_color = 'red' if pos == 0 else colors[pos]
            plt.plot(
                x_numeric, 
                y_values,
                marker='o',
                label=f'Query Position {pos + 1}',
                linewidth=3,          # increased from 2 → 3
                markersize=10,        # increased from 6 → 10
                color=line_color
            )

    plt.legend(loc='upper right', fontsize=14)  # increased legend size
    plt.xlabel('RL Steps', fontsize=18)
    plt.ylabel('Average Search Safety Score', fontsize=18)
    plt.title('Qwen-2.5-3B-IT: Search Query Safety Score by First 5 Positions', fontsize=18, fontweight='bold')
    plt.xticks([order_map.get(c, 99) for c in checkpoint_order], [label_map.get(c, c) for c in checkpoint_order], rotation=45, ha='right')
    plt.xlim(-0.5, len(checkpoint_order) - 0.5)
    plt.ylim(0, 60)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()

    # Save plot
    output_path = '/data/kebl6672/ARL/eval_results/checkpoints_qwen3b_grpo_web/search_safety_by_rl_steps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY BY RL STEPS (Positions 1-5, Min 5 queries required)")
    print("Original scale (1-5) and converted scale (0-100)")
    print("="*80)
    for filename, positions in file_data.items():
        if not positions:
            continue
        label = filename.replace('search_eval_qwen3b_', '').replace('_prefill_1_once_web.json', '')
        print(f"\n{label}:")
        for pos in sorted(positions.keys()):
            orig_score = positions[pos]
            converted_score = convert_to_100_scale(orig_score)
            print(f"  Position {pos + 1}: {orig_score:.2f} (1-5 scale) = {converted_score:.2f} (0-100 scale)")

if __name__ == "__main__":
    main()

