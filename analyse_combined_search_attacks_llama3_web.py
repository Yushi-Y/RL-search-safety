#!/usr/bin/env python3
"""
Combined analysis script to compare different search attack strategies for LLAMA3 WEB datasets:
- IT search (llama3 web) (2 positions)
- Base search (llama3 web) (2 positions) 
- Search attack average (llama3 web) (2 positions) - average of middle lines from search datasets
- Multi-search attack average (llama3 web) (2 positions) - average of all lines from multi-search datasets
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
    # File paths for search datasets (first 2 positions) - LLAMA3 WEB ONLY
    search_datasets = {
        'search_web': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_web/search_eval_llama_refusal_full_search_web.json",
        'base_web': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_web/search_eval_llama_refusal_full_search_base_web.json",
        'prefill_1_once_web': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_web/search_eval_llama_refusal_full_search_prefill_1_once_web.json",
        'prefill_2_once_web': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_web/search_eval_llama_refusal_full_search_prefill_2_once_web.json",
        'prefill_3_once_web': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_web/search_eval_llama_refusal_full_search_prefill_3_once_web.json",
        'prompt_1_web': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_web/search_eval_llama_refusal_full_search_prompt_1_web.json",
        'prompt_2_web': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_web/search_eval_llama_refusal_full_search_prompt_2_web.json"
    }
    
    # File paths for multi-search datasets (up to 2 positions) - LLAMA3 WEB ONLY
    multi_search_datasets = {
        'search_web': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_web/search_eval_llama_refusal_full_search_web.json",
        'base_web': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_web/search_eval_llama_refusal_full_search_base_web.json",
        'prompt_1_ten_searches_web': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_web/search_eval_llama_refusal_full_search_prompt_1_ten_searches_web.json",
        'prompt_2_ten_searches_web': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_web/search_eval_llama_refusal_full_search_prompt_2_ten_searches_web.json",
        'prefill_1_loop_answer_web': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_web/search_eval_llama_refusal_full_search_prefill_1_loop_answer_web.json",
        'prefill_2_loop_answer_web': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_web/search_eval_llama_refusal_full_search_prefill_2_loop_answer_web.json",
        'prefill_3_loop_answer_web': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_web/search_eval_llama_refusal_full_search_prefill_3_loop_answer_web.json"
    }
    
    print("Loading llama3 web search datasets...")
    
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
    
    print("\nLoading llama3 web multi-search datasets...")
    
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
    search_attack_datasets = ['prefill_1_once_web', 'prefill_2_once_web', 'prefill_3_once_web', 'prompt_1_web', 'prompt_2_web']
    search_attack_avg = compute_average_across_datasets(search_position_avgs, search_attack_datasets, max_positions=2)
    
    # Compute multi-search attack average (all lines from multi-search datasets)
    multi_search_attack_datasets = ['search_web', 'base_web', 'prompt_1_ten_searches_web', 'prompt_2_ten_searches_web', 
                                   'prefill_1_loop_answer_web', 'prefill_2_loop_answer_web', 'prefill_3_loop_answer_web']
    multi_search_attack_avg = compute_average_across_datasets(multi_search_position_avgs, multi_search_attack_datasets, max_positions=2)
    
    # Get IT search and base search (2 positions each)
    it_search_avg = search_position_avgs.get('search_web', {})
    base_search_avg = search_position_avgs.get('base_web', {})
    
    # Print statistics
    print(f"\n" + "="*80)
    print("COMBINED SEARCH ATTACK ANALYSIS - LLAMA3 WEB DATASETS")
    print("="*80)
    
    print(f"\nIT Search (llama3 web) (1 position):")
    if 1 in it_search_avg:
        print(f"  Position 1: {it_search_avg[1]:.3f}")
    
    print(f"\nBase Search (llama3 web) (1 position):")
    if 1 in base_search_avg:
        print(f"  Position 1: {base_search_avg[1]:.3f}")
    
    print(f"\nSearch Attack Average (llama3 web) (2 positions):")
    for pos in range(1, 3):
        if pos in search_attack_avg:
            print(f"  Position {pos}: {search_attack_avg[pos]:.3f}")
    
    print(f"\nMulti-Search Attack Average (llama3 web) (2 positions):")
    for pos in range(1, 3):
        if pos in multi_search_attack_avg:
            print(f"  Position {pos}: {multi_search_attack_avg[pos]:.3f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Function to scale harm scores: 1=0, 5=100, scale linearly
    def scale_harm_score(score):
        if score == 0:
            return 0
        return ((score - 1) / (5 - 1)) * 100
    
    # Plot IT Search (llama3 web) (1 position only)
    if it_search_avg:
        it_positions = [1]  # Only position 1
        it_means = [it_search_avg.get(1, 0)]
        it_scaled = [scale_harm_score(mean) for mean in it_means]
        plt.plot(it_positions, it_scaled, 'o-', color='#228B22', linewidth=3, markersize=10)
    
    # Plot Base Search (llama3 web) (1 position only)
    if base_search_avg:
        base_positions = [1]  # Only position 1
        base_means = [base_search_avg.get(1, 0)]
        base_scaled = [scale_harm_score(mean) for mean in base_means]
        plt.plot(base_positions, base_scaled, 'X-', color='#DC143C', linewidth=3, markersize=10)
    
    # Plot Search Attack Average (llama3 web) (1 position only)
    if search_attack_avg:
        search_attack_positions = [1]  # Only position 1
        search_attack_means = [search_attack_avg.get(1, 0)]
        search_attack_scaled = [scale_harm_score(mean) for mean in search_attack_means]
        plt.plot(search_attack_positions, search_attack_scaled, 's-', color='#6BAED6', linewidth=3, markersize=10)
    
    # Plot Multi-Search Attack Average (llama3 web) (1 position only)
    if multi_search_attack_avg:
        multi_search_positions = [1]  # Only position 1
        multi_search_means = [multi_search_attack_avg.get(1, 0)]
        multi_search_scaled = [scale_harm_score(mean) for mean in multi_search_means]
        plt.plot(multi_search_positions, multi_search_scaled, '^-', color='#FDAE6B', linewidth=3, markersize=10)
    
    plt.xlabel('Search query position', fontsize=22)
    # plt.ylabel('Average search safety', fontsize=22)  # Y-axis label removed
    # plt.legend(loc='upper right', fontsize=20, framealpha=0.9)  # Legend removed
    # plt.grid(True, alpha=0.3)  # Grid removed
    plt.xlim(0.5, 1.5)
    plt.xticks([1], fontsize=20)
    # plt.yticks(fontsize=20)  # Y-axis ticks removed
    
    # Set y-axis limits to show 0-100 scale
    plt.ylim(-5, 105)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "/data/kebl6672/AGENTIC-RL/figures/combined_search_attacks_llama3_web_position_1_only.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined search attacks comparison plot (llama3 web) saved to: {output_path}")
    
    # Save detailed results
    results = {
        'it_search_llama3_web': it_search_avg,
        'base_search_llama3_web': base_search_avg,
        'search_attack_average_llama3_web': search_attack_avg,
        'multi_search_attack_average_llama3_web': multi_search_attack_avg,
        'search_attack_datasets_llama3_web': search_attack_datasets,
        'multi_search_attack_datasets_llama3_web': multi_search_attack_datasets
    }
    
    results_path = "/data/kebl6672/AGENTIC-RL/figures/combined_search_attacks_llama3_web_position_1_only_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {results_path}")
    
    plt.show()

if __name__ == "__main__":
    main()