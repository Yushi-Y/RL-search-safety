#!/usr/bin/env python3
"""
Script to remove entries with "query": "query" from search_eval JSON files.
"""

import json
import os
import glob
from pathlib import Path

def process_file(file_path):
    """Process a single JSON file to remove entries with query='query'."""
    print(f"Processing: {file_path}")
    
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Track changes
        total_removed = 0
        questions_modified = 0
        
        # Process each question in the data
        for question_data in data:
            if 'query_evaluations' in question_data:
                original_count = len(question_data['query_evaluations'])
                
                # Filter out entries where query is "query"
                question_data['query_evaluations'] = [
                    entry for entry in question_data['query_evaluations']
                    if entry.get('query') != 'query'
                ]
                
                removed_count = original_count - len(question_data['query_evaluations'])
                if removed_count > 0:
                    total_removed += removed_count
                    questions_modified += 1
                    
                    # Recalculate average_harm_score if it exists
                    if 'average_harm_score' in question_data and question_data['query_evaluations']:
                        scores = [entry.get('harm_score', 0) for entry in question_data['query_evaluations']]
                        question_data['average_harm_score'] = sum(scores) / len(scores)
                    elif not question_data['query_evaluations']:
                        # If no queries left, remove the average_harm_score
                        question_data.pop('average_harm_score', None)
        
        # Write the modified data back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  Removed {total_removed} entries from {questions_modified} questions")
        return total_removed, questions_modified
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0

def main():
    """Main function to process all search_eval files."""
    # Directory containing the files
    base_dir = "/data/kebl6672/AGENTIC-RL/eval_results/llama3b_ppo_local"
    
    # Find all search_eval JSON files
    pattern = os.path.join(base_dir, "search_eval*.json")
    files = glob.glob(pattern)
    
    if not files:
        print("No search_eval files found!")
        return
    
    print(f"Found {len(files)} search_eval files to process")
    
    total_removed = 0
    total_questions_modified = 0
    files_processed = 0
    
    for file_path in files:
        removed, questions_modified = process_file(file_path)
        total_removed += removed
        total_questions_modified += questions_modified
        files_processed += 1
    
    print(f"\nSummary:")
    print(f"Files processed: {files_processed}")
    print(f"Total entries removed: {total_removed}")
    print(f"Total questions modified: {total_questions_modified}")

if __name__ == "__main__":
    main()