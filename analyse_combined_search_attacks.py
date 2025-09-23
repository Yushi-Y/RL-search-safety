#!/usr/bin/env python3
"""
Combined analysis script to compare different search attack strategies:
- IT search (5 positions)
- Base search (5 positions) 
- Search attack average (5 positions) - average of middle lines from search datasets
- Multi-search attack average (10 positions) - average of all lines from multi-search datasets
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_json_data(filepath):
    """Load JSON data from file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_search_harm_scores(data, dataset_name):
    """
    Extract search query harm scores from the data
    All datasets have the same structure: search_queries array and query_evaluations with harm_score
    """
    question_data = []  # List of dicts with question info and search queries
    
    for item in data:
        if 'search_queries' in item and 'query_evaluations' in item:
            search_queries = item['search_queries']
            query_scores = []
            
            # Get harm scores for each query from query_evaluations
            for eval_item in item['query_evaluations']:
                if 'harm_score' in eval_item:
                    harm_score = eval_item['harm_score']
                    # Only add numeric harm scores, skip error messages
                    if isinstance(harm_score, (int, float)) and not isinstance(harm_score, str):
                        query_scores.append(harm_score)
            
            if query_scores and len(query_scores) == len(search_queries):
                # Calculate average harm score for this question
                avg_score = sum(query_scores) / len(query_scores)
                question_data.append({
                    'question_index': item.get('question_index', len(question_data)),
                    'search_queries': search_queries,
                    'query_scores': query_scores,
                    'average_harm_score': avg_score,
                    'num_queries': len(search_queries)
                })
    
    return question_data

def analyze_query_positions(question_data, dataset_name):
    """
    Analyze harm scores by query position across all questions
    Returns dict with position -> list of harm scores
    """
    position_scores = {}  # position -> list of harm scores
    
    for question in question_data:
        query_scores = question['query_scores']
        for i, score in enumerate(query_scores):
            position = i + 1  # 1-indexed position
            if position not in position_scores:
                position_scores[position] = []
            position_scores[position].append(score)
    
    # Calculate averages for each position
    position_averages = {}
    position_counts = {}
    for position, scores in position_scores.items():
        position_averages[position] = np.mean(scores)
        position_counts[position] = len(scores)
    
    return position_averages, position_counts, position_scores

def compute_average_across_datasets(position_avgs_dict, dataset_names, max_positions=None):
    """
    Compute average harm scores across multiple datasets for each position
    """
    all_positions = set()
    for dataset_name in dataset_names:
        if dataset_name in position_avgs_dict:
            all_positions.update(position_avgs_dict[dataset_name].keys())
    
    if max_positions:
        all_positions = [pos for pos in all_positions if pos <= max_positions]
    else:
        all_positions = sorted(all_positions)
    
    averaged_scores = {}
    for position in all_positions:
        scores_at_position = []
        for dataset_name in dataset_names:
            if dataset_name in position_avgs_dict and position in position_avgs_dict[dataset_name]:
                scores_at_position.append(position_avgs_dict[dataset_name][position])
        
        if scores_at_position:
            averaged_scores[position] = np.mean(scores_at_position)
    
    return averaged_scores

def main():
    # File paths for search datasets (first 5 positions) - LOCAL ONLY
    search_datasets = {
        'search': "/data/kebl6672/AGENTIC-RL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search.json",
        'base': "/data/kebl6672/AGENTIC-RL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_base.json",
        'prefill_1_once': "/data/kebl6672/AGENTIC-RL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_prefill_1_once.json",
        'prefill_2_once': "/data/kebl6672/AGENTIC-RL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_prefill_2_once.json",
        'prefill_3_once': "/data/kebl6672/AGENTIC-RL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_prefill_3_once.json",
        'prompt_1': "/data/kebl6672/AGENTIC-RL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_prompt_1.json",
        'prompt_2': "/data/kebl6672/AGENTIC-RL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_prompt_2.json"
    }
    
    # File paths for multi-search datasets (up to 10 positions)
    multi_search_datasets = {
        'search': "/data/kebl6672/AGENTIC-RL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search.json",
        'base': "/data/kebl6672/AGENTIC-RL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_base.json",
        'prompt_1_ten_searches': "/data/kebl6672/AGENTIC-RL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_prompt_1_ten_searches.json",
        'prompt_2_ten_searches': "/data/kebl6672/AGENTIC-RL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_prompt_2_ten_searches.json",
        'prefill_1_loop_answer': "/data/kebl6672/AGENTIC-RL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_prefill_1_loop_answer.json",
        'prefill_2_loop_answer': "/data/kebl6672/AGENTIC-RL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_prefill_2_loop_answer.json",
        'prefill_3_loop_answer': "/data/kebl6672/AGENTIC-RL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_prefill_3_loop_answer.json"
    }
    
    print("Loading search datasets...")
    
    # Load data for search datasets
    search_data = {}
    search_question_data = {}
    search_position_avgs = {}
    search_position_counts = {}
    
    for dataset_name, filepath in search_datasets.items():
        print(f"Loading {dataset_name}...")
        try:
            data = load_json_data(filepath)
            search_data[dataset_name] = data
            print(f"  {dataset_name}: {len(data)} questions")
            
            # Extract search query data
            question_data = extract_search_harm_scores(data, dataset_name)
            search_question_data[dataset_name] = question_data
            print(f"  {dataset_name}: {len(question_data)} questions with search queries")
            
            # Analyze query positions
            position_avgs, position_counts, position_scores = analyze_query_positions(question_data, dataset_name)
            search_position_avgs[dataset_name] = position_avgs
            search_position_counts[dataset_name] = position_counts
            
            print(f"  {dataset_name} query positions found: {sorted(position_avgs.keys())}")
            
        except FileNotFoundError:
            print(f"  {dataset_name}: File not found - {filepath}")
            continue
        except Exception as e:
            print(f"  {dataset_name}: Error loading - {e}")
            continue
    
    print("\nLoading multi-search datasets...")
    
    # Load data for multi-search datasets
    multi_search_data = {}
    multi_search_question_data = {}
    multi_search_position_avgs = {}
    multi_search_position_counts = {}
    
    for dataset_name, filepath in multi_search_datasets.items():
        print(f"Loading {dataset_name}...")
        try:
            data = load_json_data(filepath)
            multi_search_data[dataset_name] = data
            print(f"  {dataset_name}: {len(data)} questions")
            
            # Extract search query data
            question_data = extract_search_harm_scores(data, dataset_name)
            multi_search_question_data[dataset_name] = question_data
            print(f"  {dataset_name}: {len(question_data)} questions with search queries")
            
            # Analyze query positions
            position_avgs, position_counts, position_scores = analyze_query_positions(question_data, dataset_name)
            multi_search_position_avgs[dataset_name] = position_avgs
            multi_search_position_counts[dataset_name] = position_counts
            
            print(f"  {dataset_name} query positions found: {sorted(position_avgs.keys())}")
            
        except FileNotFoundError:
            print(f"  {dataset_name}: File not found - {filepath}")
            continue
        except Exception as e:
            print(f"  {dataset_name}: Error loading - {e}")
            continue
    
    # Compute search attack average (middle lines from search datasets, excluding IT search and base search)
    search_attack_datasets = ['prefill_1_once', 'prefill_2_once', 'prefill_3_once', 'prompt_1', 'prompt_2']
    search_attack_avg = compute_average_across_datasets(search_position_avgs, search_attack_datasets, max_positions=5)
    
    # Compute multi-search attack average (all lines from multi-search datasets)
    multi_search_attack_datasets = ['search', 'base', 'prompt_1_ten_searches', 'prompt_2_ten_searches', 
                                   'prefill_1_loop_answer', 'prefill_2_loop_answer', 'prefill_3_loop_answer']
    multi_search_attack_avg = compute_average_across_datasets(multi_search_position_avgs, multi_search_attack_datasets, max_positions=10)
    
    # Get IT search and base search (5 positions each)
    it_search_avg = search_position_avgs.get('search', {})
    base_search_avg = search_position_avgs.get('base', {})
    
    # Print statistics
    print(f"\n" + "="*80)
    print("COMBINED SEARCH ATTACK ANALYSIS")
    print("="*80)
    
    print(f"\nIT Search (5 positions):")
    for pos in range(1, 6):
        if pos in it_search_avg:
            print(f"  Position {pos}: {it_search_avg[pos]:.3f}")
    
    print(f"\nBase Search (5 positions):")
    for pos in range(1, 6):
        if pos in base_search_avg:
            print(f"  Position {pos}: {base_search_avg[pos]:.3f}")
    
    print(f"\nSearch Attack Average (5 positions):")
    for pos in range(1, 6):
        if pos in search_attack_avg:
            print(f"  Position {pos}: {search_attack_avg[pos]:.3f}")
    
    print(f"\nMulti-Search Attack Average (10 positions):")
    for pos in range(1, 11):
        if pos in multi_search_attack_avg:
            print(f"  Position {pos}: {multi_search_attack_avg[pos]:.3f}")
    
    # Create visualization
    plt.figure(figsize=(16, 10))
    
    # Function to scale harm scores: 1=0, 5=100, scale linearly
    def scale_harm_score(score):
        if score == 0:
            return 0
        return ((score - 1) / (5 - 1)) * 100
    
    # Plot IT Search (5 positions)
    if it_search_avg:
        it_positions = list(range(1, 6))
        it_means = [it_search_avg.get(pos, 0) for pos in it_positions]
        it_scaled = [scale_harm_score(mean) for mean in it_means]
        plt.plot(it_positions, it_scaled, 'o-', label='IT Search', color='#228B22', linewidth=3, markersize=10)
        
        # Add value labels
        for pos, mean_val, scaled_val in zip(it_positions, it_means, it_scaled):
            if mean_val > 0:
                plt.annotate(f'{scaled_val:.0f}', (pos, scaled_val), 
                           textcoords="offset points", xytext=(0, 8), 
                           ha='center', fontsize=16, color='#228B22', fontweight='bold')
    
    # Plot Base Search (5 positions)
    if base_search_avg:
        base_positions = list(range(1, 6))
        base_means = [base_search_avg.get(pos, 0) for pos in base_positions]
        base_scaled = [scale_harm_score(mean) for mean in base_means]
        plt.plot(base_positions, base_scaled, 'X-', label='Base Search', color='#DC143C', linewidth=3, markersize=10)
        
        # Add value labels
        for pos, mean_val, scaled_val in zip(base_positions, base_means, base_scaled):
            if mean_val > 0:
                plt.annotate(f'{scaled_val:.0f}', (pos, scaled_val), 
                           textcoords="offset points", xytext=(0, 8), 
                           ha='center', fontsize=16, color='#DC143C', fontweight='bold')
    
    # Plot Search Attack Average (5 positions)
    if search_attack_avg:
        search_attack_positions = list(range(1, 6))
        search_attack_means = [search_attack_avg.get(pos, 0) for pos in search_attack_positions]
        search_attack_scaled = [scale_harm_score(mean) for mean in search_attack_means]
        plt.plot(search_attack_positions, search_attack_scaled, 's-', label='Search Attack', color='#6BAED6', linewidth=3, markersize=10)
        
        # Add value labels
        for pos, mean_val, scaled_val in zip(search_attack_positions, search_attack_means, search_attack_scaled):
            if mean_val > 0:
                plt.annotate(f'{scaled_val:.0f}', (pos, scaled_val), 
                           textcoords="offset points", xytext=(0, 8), 
                           ha='center', fontsize=16, color='#6BAED6', fontweight='bold')
    
    # Plot Multi-Search Attack Average (5 positions only)
    if multi_search_attack_avg:
        multi_search_positions = list(range(1, 6))  # Only first 5 positions
        multi_search_means = [multi_search_attack_avg.get(pos, 0) for pos in multi_search_positions]
        multi_search_scaled = [scale_harm_score(mean) for mean in multi_search_means]
        plt.plot(multi_search_positions, multi_search_scaled, '^-', label='Multi-Search Attack', color='#FDAE6B', linewidth=3, markersize=10)
        
        # Add value labels
        for pos, mean_val, scaled_val in zip(multi_search_positions, multi_search_means, multi_search_scaled):
            if mean_val > 0:
                plt.annotate(f'{scaled_val:.0f}', (pos, scaled_val), 
                           textcoords="offset points", xytext=(0, 8), 
                           ha='center', fontsize=16, color='#FDAE6B', fontweight='bold')
    
    plt.xlabel('Search query position', fontsize=22)
    plt.ylabel('Average search safety', fontsize=22)
    plt.legend(loc='upper right', fontsize=20, framealpha=0.9)
    # plt.grid(True, alpha=0.3)  # Grid removed
    plt.xlim(0.5, 5.5)
    plt.xticks(range(1, 6), fontsize=20)
    plt.yticks(fontsize=20)
    
    # Set y-axis limits to show 0-100 scale
    plt.ylim(-5, 105)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "/data/kebl6672/AGENTIC-RL/figures/combined_search_attacks_first_5_positions.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined search attacks comparison plot saved to: {output_path}")
    
    # Save detailed results
    results = {
        'it_search': it_search_avg,
        'base_search': base_search_avg,
        'search_attack_average': search_attack_avg,
        'multi_search_attack_average': multi_search_attack_avg,
        'search_attack_datasets': search_attack_datasets,
        'multi_search_attack_datasets': multi_search_attack_datasets
    }
    
    results_path = "/data/kebl6672/AGENTIC-RL/figures/combined_search_attacks_first_5_positions_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {results_path}")
    
    plt.show()

if __name__ == "__main__":
    main()