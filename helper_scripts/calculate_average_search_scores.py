#!/usr/bin/env python3
"""
Script to calculate average harm scores for all queries in each search_eval JSON file.
"""

import json
import os
import glob
from pathlib import Path

def calculate_file_average(file_path):
    """Calculate the average harm score for all queries in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_harm_scores = []
        total_entries = 0
        error_entries = 0
        
        # Collect all harm scores from all questions
        for question_data in data:
            if 'query_evaluations' in question_data:
                for entry in question_data['query_evaluations']:
                    if 'harm_score' in entry:
                        total_entries += 1
                        harm_score = entry['harm_score']
                        # Only include numeric harm scores (skip error messages)
                        if isinstance(harm_score, (int, float)) and not isinstance(harm_score, bool):
                            all_harm_scores.append(harm_score)
                        else:
                            error_entries += 1
        
        if not all_harm_scores:
            return None, 0, total_entries, error_entries
        
        average_score = sum(all_harm_scores) / len(all_harm_scores)
        return average_score, len(all_harm_scores), total_entries, error_entries
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, 0, 0, 0

def main():
    """Main function to calculate averages for all search_eval files."""
    # Directory containing the files
    base_dir = "/data/kebl6672/AGENTIC-RL/eval_results"
    
    # Find all search_eval JSON files in all subdirectories
    pattern = os.path.join(base_dir, "**", "search_eval*.json")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print("No search_eval files found!")
        return
    
    print(f"Found {len(files)} search_eval files to process")
    
    # Organize files by folder
    folder_results = {}
    all_results = []
    
    for file_path in sorted(files):
        # Get relative path and folder
        rel_path = os.path.relpath(file_path, base_dir)
        folder = os.path.dirname(rel_path)
        filename = os.path.basename(rel_path)
        
        average_score, num_queries, total_entries, error_entries = calculate_file_average(file_path)
        
        if average_score is not None:
            result = {
                'file': filename,
                'full_path': rel_path,
                'average_score': average_score,
                'num_queries': num_queries,
                'total_entries': total_entries,
                'error_entries': error_entries
            }
            all_results.append(result)
            
            if folder not in folder_results:
                folder_results[folder] = []
            folder_results[folder].append(result)
    
    # Print results organized by folder
    print("\n" + "="*100)
    print("AVERAGE HARM SCORES BY FOLDER")
    print("="*100)
    
    for folder in sorted(folder_results.keys()):
        print(f"\n📁 {folder}/")
        print("-" * 80)
        
        folder_scores = []
        folder_queries = 0
        
        for result in sorted(folder_results[folder], key=lambda x: x['file']):
            error_info = f" (Errors: {result['error_entries']})" if result['error_entries'] > 0 else ""
            print(f"  {result['file']:<50} | Avg: {result['average_score']:.3f} | Queries: {result['num_queries']}{error_info}")
            folder_scores.append(result['average_score'])
            folder_queries += result['num_queries']
        
        # Folder summary
        if folder_scores:
            folder_avg = sum(folder_scores) / len(folder_scores)
            print(f"  {'FOLDER SUMMARY':<50} | Avg: {folder_avg:.3f} | Queries: {folder_queries}")
    
    # Overall statistics
    if all_results:
        all_scores = [r['average_score'] for r in all_results]
        total_queries = sum(r['num_queries'] for r in all_results)
        total_entries = sum(r['total_entries'] for r in all_results)
        total_errors = sum(r['error_entries'] for r in all_results)
        
        print("\n" + "="*100)
        print("OVERALL STATISTICS")
        print("="*100)
        print(f"Total files processed: {len(all_results)}")
        print(f"Total queries processed: {total_queries}")
        print(f"Total entries found: {total_entries}")
        print(f"Error entries (skipped): {total_errors}")
        print(f"Overall average across all files: {sum(all_scores) / len(all_scores):.3f}")
        
        # Show min and max
        min_file = min(all_results, key=lambda x: x['average_score'])
        max_file = max(all_results, key=lambda x: x['average_score'])
        
        print(f"Lowest average: {min_file['average_score']:.3f} ({min_file['full_path']})")
        print(f"Highest average: {max_file['average_score']:.3f} ({max_file['full_path']})")
        
        # Folder averages
        print(f"\nFolder averages:")
        for folder in sorted(folder_results.keys()):
            folder_scores = [r['average_score'] for r in folder_results[folder]]
            folder_avg = sum(folder_scores) / len(folder_scores)
            print(f"  {folder:<30} | Avg: {folder_avg:.3f} | Files: {len(folder_scores)}")

if __name__ == "__main__":
    main()