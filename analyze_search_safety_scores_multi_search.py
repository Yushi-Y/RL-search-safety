#!/usr/bin/env python3
"""
Multi-dataset analysis script to compare average harm scores for search queries between datasets:
- search (IT) - plot first 5 positions
- base - plot first 5 positions
- prompt_1_ten_searches - plot first 5 positions
- prompt_2_ten_searches - plot first 5 positions
- prefill_1_loop_answer (plot all 10 positions)
- prefill_2_loop_answer (plot all 10 positions)
- prefill_3_loop_answer (plot all 10 positions)
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

def count_questions_by_query_count(question_data):
    """Count number of questions with different numbers of search queries"""
    query_count_distribution = {}
    for question in question_data:
        num_queries = question['num_queries']
        if num_queries not in query_count_distribution:
            query_count_distribution[num_queries] = 0
        query_count_distribution[num_queries] += 1
    
    return query_count_distribution

def main():
    # File paths for all datasets
    datasets = {
        'search': "/data/kebl6672/ARL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search.json",
        'base': "/data/kebl6672/ARL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_base.json",
        'prompt_1_ten_searches': "/data/kebl6672/ARL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_prompt_1_ten_searches.json",
        'prompt_2_ten_searches': "/data/kebl6672/ARL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_prompt_2_ten_searches.json",
        'prefill_1_loop_answer': "/data/kebl6672/ARL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_prefill_1_loop_answer.json",
        'prefill_2_loop_answer': "/data/kebl6672/ARL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_prefill_2_loop_answer.json",
        'prefill_3_loop_answer': "/data/kebl6672/ARL/eval_results/qwen7b_ppo_local/search_eval_qwen_refusal_full_search_prefill_3_loop_answer.json"
    }
    
    print("Loading data...")
    
    # Load data for all datasets
    all_data = {}
    all_question_data = {}
    all_position_avgs = {}
    all_position_counts = {}
    all_query_count_dist = {}
    
    for dataset_name, filepath in datasets.items():
        print(f"Loading {dataset_name}...")
        try:
            data = load_json_data(filepath)
            all_data[dataset_name] = data
            print(f"  {dataset_name}: {len(data)} questions")
            
            # Extract search query data
            question_data = extract_search_harm_scores(data, dataset_name)
            all_question_data[dataset_name] = question_data
            print(f"  {dataset_name}: {len(question_data)} questions with search queries")
            
            # Analyze query positions
            position_avgs, position_counts, position_scores = analyze_query_positions(question_data, dataset_name)
            all_position_avgs[dataset_name] = position_avgs
            all_position_counts[dataset_name] = position_counts
            
            # Count questions by query count
            query_count_dist = count_questions_by_query_count(question_data)
            all_query_count_dist[dataset_name] = query_count_dist
            
            print(f"  {dataset_name} query positions found: {sorted(position_avgs.keys())}")
            
        except FileNotFoundError:
            print(f"  {dataset_name}: File not found - {filepath}")
            continue
        except Exception as e:
            print(f"  {dataset_name}: Error loading - {e}")
            continue
    
    # Print position statistics for all datasets
    print(f"\n" + "="*80)
    print("HARM SCORES BY QUERY POSITION")
    print("="*80)
    
    for dataset_name in all_position_avgs.keys():
        print(f"\n{dataset_name.upper()}:")
        print(f"  - Number of questions with search queries: {len(all_question_data[dataset_name])}")
        print(f"  - Average harm scores by position:")
        for pos in sorted(all_position_avgs[dataset_name].keys()):
            print(f"    Position {pos}: {all_position_avgs[dataset_name][pos]:.3f} (n={all_position_counts[dataset_name][pos]})")
    
    # Print query count distribution for all datasets
    print(f"\n" + "="*80)
    print("QUESTIONS BY NUMBER OF SEARCH QUERIES")
    print("="*80)
    
    for dataset_name in all_query_count_dist.keys():
        print(f"\n{dataset_name.upper()}:")
        print(f"  - Total questions with search queries: {len(all_question_data[dataset_name])}")
        print(f"  - Questions by query count:")
        for num_queries in sorted(all_query_count_dist[dataset_name].keys()):
            count = all_query_count_dist[dataset_name][num_queries]
            percentage = (count / len(all_question_data[dataset_name])) * 100
            print(f"    {num_queries} queries: {count} questions ({percentage:.1f}%)")
    
    # Create visualization - Line plot
    plt.figure(figsize=(16, 10))
    
    # Define colors and markers for each dataset
    colors = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4', '#9467bd', '#17becf', '#e377c2', '#8c564b', '#bcbd22']
    markers = ['o', 'X', 's', '^', 'D', 'v', 'p', 'h', '*']
    
    # Function to scale harm scores: 1=0, 5=100, scale linearly
    def scale_harm_score(score):
        if score == 0:
            return 0
        return ((score - 1) / (5 - 1)) * 100
    
    # Plot search/base/prompt datasets (first 5 positions) and prefill datasets (all 10 positions)
    search_positions = list(range(1, 6))
    prefill_positions = list(range(1, 11))
    
    # Plot each dataset
    for i, (dataset_name, position_avgs) in enumerate(all_position_avgs.items()):
        if dataset_name in all_position_avgs:
            # Determine which positions to plot based on dataset type
            if dataset_name in ['search', 'base', 'prompt_1_ten_searches', 'prompt_2_ten_searches']:
                positions_to_plot = search_positions
            else:  # prefill datasets
                positions_to_plot = prefill_positions
            
            means = [position_avgs.get(pos, 0) for pos in positions_to_plot]
            scaled_means = [scale_harm_score(mean) for mean in means]
            
            # Assign colors based on dataset type
            if dataset_name == 'search':
                color = colors[0]  # Green for IT search
            elif dataset_name == 'base':
                color = '#808080'  # Light gray for Base
            elif dataset_name in ['prompt_1_ten_searches', 'prompt_2_ten_searches']:
                color = '#6BAED6'  # Soft blue for both Prompt A and B
            elif dataset_name in ['prefill_1_loop_answer', 'prefill_2_loop_answer']:
                color = '#FDAE6B'  # Soft orange for both Prefill A and B
            elif dataset_name == 'prefill_3_loop_answer':
                color = '#BC80BD'  # Soft purple for Prefill C
            else:
                color = colors[i % len(colors)]
            
            marker = markers[i % len(markers)]
            
            # Clean up dataset names for legend
            if dataset_name == 'search':
                clean_name = 'IT search'
            elif dataset_name == 'base':
                clean_name = 'Base search'
            elif dataset_name == 'prompt_1_ten_searches':
                clean_name = 'Prompt A (10 times)'
            elif dataset_name == 'prompt_2_ten_searches':
                clean_name = 'Prompt B (10 times)'
            elif dataset_name == 'prefill_1_loop_answer':
                clean_name = 'Prefill A (10 times)'
            elif dataset_name == 'prefill_2_loop_answer':
                clean_name = 'Prefill B (10 times)'
            elif dataset_name == 'prefill_3_loop_answer':
                clean_name = 'Prefill C (10 times)'
            else:
                clean_name = dataset_name.replace('_', ' ').title()
            
            # Only plot if we have data for at least one position
            if any(mean > 0 for mean in means):
                plt.plot(positions_to_plot, scaled_means, f'{marker}-', 
                        label=clean_name, 
                        color=color, linewidth=2, markersize=8)
                
                # Add value labels on points - removed all numbers
                # for j, (pos, mean_val, scaled_val) in enumerate(zip(positions_to_plot, means, scaled_means)):
                #     if mean_val > 0:
                #         plt.annotate(f'{scaled_val:.0f}', (pos, scaled_val), 
                #                    textcoords="offset points", 
                #                    xytext=(0, 12 + (i * 3)),  # Offset labels to avoid overlap
                #                    ha='center', fontsize=13, color=color, fontweight='bold')
    
    # plt.xlabel('Search query position', fontsize=20)  # X-axis label removed
    plt.ylabel('Average search safety', fontsize=20)
    # Remove title
    plt.legend(loc='lower right', fontsize=16, framealpha=0.9)
    # plt.grid(True, alpha=0.3, axis='y')  # Grid removed
    plt.xlim(0.5, 10.5)
    plt.xticks(list(range(1, 11)), fontsize=18)
    plt.yticks(fontsize=18)
    
    # Set y-axis limits to show 0-100 scale
    plt.ylim(-5, 105)
    
    # Add vertical line to separate search/base/prompt and prefill regions
    plt.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)
    plt.text(3, 95, 'IT-search/Base-search/Prompt (10 times)\n1-5 queries', ha='center', fontsize=17)
    plt.text(8, 95, 'Prefill (10 times)\n1-10 queries', ha='center', fontsize=17)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "/data/kebl6672/ARL/figures/multi_search_safety_scores_10_positions.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSearch safety scores line plot saved to: {output_path}")
    
    # Create a comparison table for all positions
    print(f"\n" + "="*80)
    print("COMPARISON TABLE - ALL POSITIONS")
    print("="*80)
    
    # Print header
    print(f"{'Position':<8}", end="")
    for dataset_name in all_position_avgs.keys():
        clean_name = dataset_name.replace('_', ' ').title()
        if len(clean_name) > 15:
            clean_name = clean_name[:12] + "..."
        print(f"{clean_name:<15}", end="")
    print()
    print("-" * (8 + 15 * len(all_position_avgs)))
    
    # Print data for each position (1-10)
    for pos in range(1, 11):
        print(f"{pos:<8}", end="")
        for dataset_name in all_position_avgs.keys():
            if pos in all_position_avgs[dataset_name]:
                avg_score = all_position_avgs[dataset_name][pos]
                count = all_position_counts[dataset_name][pos]
                print(f"{avg_score:.3f} (n={count})<15", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()
    
    # Create query count distribution comparison table
    print(f"\n" + "="*80)
    print("QUERY COUNT DISTRIBUTION COMPARISON TABLE")
    print("="*80)
    
    # Print header
    print(f"{'Queries':<8}", end="")
    for dataset_name in all_query_count_dist.keys():
        clean_name = dataset_name.replace('_', ' ').title()
        if len(clean_name) > 15:
            clean_name = clean_name[:12] + "..."
        print(f"{clean_name:<15}", end="")
    print()
    print("-" * (8 + 15 * len(all_query_count_dist)))
    
    # Print data for each query count (1-10)
    for num_queries in range(1, 11):
        print(f"{num_queries:<8}", end="")
        for dataset_name in all_query_count_dist.keys():
            if num_queries in all_query_count_dist[dataset_name]:
                count = all_query_count_dist[dataset_name][num_queries]
                total = len(all_question_data[dataset_name])
                percentage = (count / total) * 100
                print(f"{count} ({percentage:.1f}%)<15", end="")
            else:
                print(f"{'0 (0.0%)':<15}", end="")
        print()
    
    # Save detailed results
    results = {}
    for dataset_name in all_position_avgs.keys():
        results[dataset_name] = {
            'question_data': all_question_data[dataset_name],
            'position_averages': all_position_avgs[dataset_name],
            'position_counts': all_position_counts[dataset_name]
        }
    
    results_path = "/data/kebl6672/ARL/figures/multi_search_safety_scores_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {results_path}")
    
    plt.show()

if __name__ == "__main__":
    main()