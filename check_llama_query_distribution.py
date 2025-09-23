#!/usr/bin/env python3
"""
Check query distribution for Llama3 local search attack datasets
"""

import json
import os

def load_json_data(filepath):
    """Load JSON data from file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_query_distribution(data, dataset_name):
    """Analyze query count distribution for a dataset"""
    query_counts = []
    
    for item in data:
        if 'search_queries' in item:
            num_queries = len(item['search_queries'])
            query_counts.append(num_queries)
    
    if query_counts:
        distribution = {}
        for count in query_counts:
            distribution[count] = distribution.get(count, 0) + 1
        
        total_questions = len(query_counts)
        print(f"\n{dataset_name}:")
        print(f"  Total questions: {total_questions}")
        print(f"  Query count distribution:")
        for count in sorted(distribution.keys()):
            percentage = (distribution[count] / total_questions) * 100
            print(f"    {count} queries: {distribution[count]} questions ({percentage:.1f}%)")
        
        avg_queries = sum(query_counts) / len(query_counts)
        print(f"  Average queries per question: {avg_queries:.2f}")
        print(f"  Min queries: {min(query_counts)}")
        print(f"  Max queries: {max(query_counts)}")
    else:
        print(f"\n{dataset_name}: No data found")

def main():
    # File paths for Llama3 local search attack datasets
    datasets = {
        'search': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_local/search_eval_llama_refusal_full_search.json",
        'base': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_local/search_eval_llama_refusal_full_search_base.json",
        'prefill_1_once': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_local/search_eval_llama_refusal_full_search_prefill_1_once.json",
        'prefill_2_once': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_local/search_eval_llama_refusal_full_search_prefill_2_once.json",
        'prefill_3_once': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_local/search_eval_llama_refusal_full_search_prefill_3_once.json",
        'prompt_1': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_local/search_eval_llama_refusal_full_search_prompt_1.json",
        'prompt_2': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_local/search_eval_llama_refusal_full_search_prompt_2.json",
        'prompt_1_ten_searches': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_local/search_eval_llama_refusal_full_search_prompt_1_ten_searches.json",
        'prompt_2_ten_searches': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_local/search_eval_llama_refusal_full_search_prompt_2_ten_searches.json",
        'prefill_1_loop_answer': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_local/search_eval_llama_refusal_full_search_prefill_1_loop_answer.json",
        'prefill_2_loop_answer': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_local/search_eval_llama_refusal_full_search_prefill_2_loop_answer.json",
        'prefill_3_loop_answer': "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_local/search_eval_llama_refusal_full_search_prefill_3_loop_answer.json"
    }
    
    print("="*80)
    print("LLAMA3 LOCAL SEARCH ATTACK QUERY DISTRIBUTION ANALYSIS")
    print("="*80)
    
    for dataset_name, filepath in datasets.items():
        try:
            data = load_json_data(filepath)
            analyze_query_distribution(data, dataset_name)
        except FileNotFoundError:
            print(f"\n{dataset_name}: File not found - {filepath}")
        except Exception as e:
            print(f"\n{dataset_name}: Error loading - {e}")

if __name__ == "__main__":
    main()