#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_file(filepath):
    """
    Analyze a single semantic similarity file.
    Returns: dict with average similarity for each query position
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Group by query_index
    query_similarities = defaultdict(list)
    
    for entry in data:
        query_idx = entry.get('query_index')
        similarity = entry.get('similarity')
        if query_idx is not None and similarity is not None:
            query_similarities[query_idx].append(similarity)
    
    # Calculate averages
    averages = {}
    for query_idx in sorted(query_similarities.keys()):
        sims = query_similarities[query_idx]
        if sims:
            averages[query_idx + 1] = sum(sims) / len(sims)  # +1 to make it 1-indexed
    
    return averages

def main():
    eval_dir = Path('/data/kebl6672/ARL/eval_results')
    
    # Define which files to process
    qwen_local_dir = eval_dir / 'qwen7b_ppo_local'
    qwen_web_dir = eval_dir / 'qwen7b_ppo_web'
    llama_local_dir = eval_dir / 'llama3b_ppo_local'
    llama_web_dir = eval_dir / 'llama3b_ppo_web'
    
    results = {}
    
    # Process Qwen Local files
    print("=" * 80)
    print("QWEN LOCAL (positions 1-5)")
    print("=" * 80)
    for file in sorted(qwen_local_dir.glob('semantic_similarity_*.json')):
        filename = file.name
        averages = analyze_file(file)
        results[f"qwen_local/{filename}"] = averages
        
        print(f"\n{filename}:")
        for pos in range(1, 6):  # positions 1-5
            if pos in averages:
                print(f"  Position {pos}: {averages[pos]:.6f}")
            else:
                print(f"  Position {pos}: N/A")
    
    # Process Qwen Web files
    print("\n" + "=" * 80)
    print("QWEN WEB (positions 1-5)")
    print("=" * 80)
    for file in sorted(qwen_web_dir.glob('semantic_similarity_*.json')):
        filename = file.name
        averages = analyze_file(file)
        results[f"qwen_web/{filename}"] = averages
        
        print(f"\n{filename}:")
        for pos in range(1, 6):  # positions 1-5
            if pos in averages:
                print(f"  Position {pos}: {averages[pos]:.6f}")
            else:
                print(f"  Position {pos}: N/A")
    
    # Process Llama Local files (position 1 only)
    print("\n" + "=" * 80)
    print("LLAMA LOCAL (position 1 only)")
    print("=" * 80)
    for file in sorted(llama_local_dir.glob('semantic_similarity_*.json')):
        filename = file.name
        averages = analyze_file(file)
        results[f"llama_local/{filename}"] = averages
        
        print(f"\n{filename}:")
        if 1 in averages:
            print(f"  Position 1: {averages[1]:.6f}")
        else:
            print(f"  Position 1: N/A")
    
    # Process Llama Web files (position 1 only)
    print("\n" + "=" * 80)
    print("LLAMA WEB (position 1 only)")
    print("=" * 80)
    for file in sorted(llama_web_dir.glob('semantic_similarity_*.json')):
        filename = file.name
        averages = analyze_file(file)
        results[f"llama_web/{filename}"] = averages
        
        print(f"\n{filename}:")
        if 1 in averages:
            print(f"  Position 1: {averages[1]:.6f}")
        else:
            print(f"  Position 1: N/A")
    
    # Save results to JSON file
    output_file = '/data/kebl6672/ARL/semantic_similarity_summary.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)

if __name__ == '__main__':
    main()

