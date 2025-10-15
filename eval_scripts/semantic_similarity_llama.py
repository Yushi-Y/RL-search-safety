import json
import time
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set cache directories to use /data partition instead of /home
os.environ['TORCH_HOME'] = '/data/kebl6672/.cache/torch'
os.environ['HF_HOME'] = '/data/kebl6672/.cache/huggingface'

# Create cache directories if they don't exist
os.makedirs('/data/kebl6672/.cache/torch', exist_ok=True)
os.makedirs('/data/kebl6672/.cache/huggingface', exist_ok=True)

def compute_semantic_similarity(questions, queries, model):
    """
    Compute semantic similarity between questions and their search queries.
    
    Args:
        questions: List of original harmful questions
        queries: List of lists of search queries (one list per question)
        model: SentenceTransformer model
    
    Returns:
        List of dictionaries containing individual query similarities
    """
    all_results = []
    
    for i, (question, query_list) in enumerate(zip(questions, queries)):
        if not query_list:
            continue
        
        # Encode question once
        question_embedding = model.encode([question])
        
        # Compute cosine similarity for each query individually
        for j, query in enumerate(query_list):
            query_embedding = model.encode([query])
            similarity = cosine_similarity(question_embedding, query_embedding)[0][0]
            
            all_results.append({
                'question_index': i,
                'question': question,
                'query': query,
                'query_index': j,
                'similarity': float(similarity)
            })
    
    return all_results

def process_single_file(input_file, output_file, model):
    """Process a single input file and save results."""
    
    print(f"Loading responses from {input_file}...")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file {input_file} not found!")
        print(f"Available files in {os.path.dirname(input_file)}/:")
        try:
            files = os.listdir(os.path.dirname(input_file))
            for f in files:
                print(f"  - {f}")
        except:
            print("  Directory not found")
        return
    
    with open(input_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} responses to evaluate")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize sentence transformer model
    print("Loading sentence transformer model...")
    print("Using model: all-mpnet-base-v2 (high accuracy, state-of-the-art)")
    print("(This model provides better semantic understanding than MiniLM)")
    model = SentenceTransformer('all-mpnet-base-v2')
    print("✓ Model loaded successfully")
    
    # Prepare data
    questions = []
    queries = []
    question_indices = []
    
    for i, item in enumerate(data):
        question = item.get("question", "")
        search_information = item.get("search_information", [])
        
        if not question:
            print(f"Skipping item {i+1}: missing question")
            continue
        
        # Extract all search queries
        search_queries = []
        for search_item in search_information:
            query = search_item.get("query", "").strip()
            if query:
                search_queries.append(query)
        
        questions.append(question)
        queries.append(search_queries)
        question_indices.append(item.get("question_index", i))
    
    print(f"\nProcessing {len(questions)} questions...")
    print("Computing semantic similarities...")
    
    # Compute similarities
    start_time = time.time()
    results = compute_semantic_similarity(questions, queries, model)
    elapsed_time = time.time() - start_time
    
    print(f"✓ Completed in {elapsed_time:.2f} seconds")
    
    # Add original question indices to results
    question_idx_map = {i: question_indices[i] for i in range(len(question_indices))}
    for result in results:
        result['original_question_index'] = question_idx_map.get(result['question_index'], result['question_index'])
    
    # Save results directly (just the list of results)
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved successfully")
    print(f"✓ Saved {len(results)} query-question similarity pairs")

def main():
    """Main function to process multiple files in batch."""
    
    # Define batch of input-output pairs
    file_pairs = [
        # LLAMA WEB
        ("all_responses/llama3b_ppo_web/llama_refusal_full_search_web.json",
         "eval_results/llama3b_ppo_web/semantic_similarity_llama_refusal_full_search_web.json"),
        
        ("all_responses/llama3b_ppo_web/llama_refusal_full_search_base_web.json",
         "eval_results/llama3b_ppo_web/semantic_similarity_llama_refusal_full_search_base_web.json"),
        
        ("all_responses/llama3b_ppo_web/llama_refusal_full_search_prefill_1_once_web.json",
         "eval_results/llama3b_ppo_web/semantic_similarity_llama_refusal_full_search_prefill_1_once_web.json"),
        
        ("all_responses/llama3b_ppo_web/llama_refusal_full_search_prefill_2_once_web.json",
         "eval_results/llama3b_ppo_web/semantic_similarity_llama_refusal_full_search_prefill_2_once_web.json"),
        
        ("all_responses/llama3b_ppo_web/llama_refusal_full_search_prefill_3_once_web.json",
         "eval_results/llama3b_ppo_web/semantic_similarity_llama_refusal_full_search_prefill_3_once_web.json"),
        
        ("all_responses/llama3b_ppo_web/llama_refusal_full_search_prompt_1_web.json",
         "eval_results/llama3b_ppo_web/semantic_similarity_llama_refusal_full_search_prompt_1_web.json"),
        
        ("all_responses/llama3b_ppo_web/llama_refusal_full_search_prompt_2_web.json",
         "eval_results/llama3b_ppo_web/semantic_similarity_llama_refusal_full_search_prompt_2_web.json"),
        
        ("all_responses/llama3b_ppo_web/llama_refusal_full_search_prompt_1_ten_searches_web.json",
         "eval_results/llama3b_ppo_web/semantic_similarity_llama_refusal_full_search_prompt_1_ten_searches_web.json"),
        
        ("all_responses/llama3b_ppo_web/llama_refusal_full_search_prompt_2_ten_searches_web.json",
         "eval_results/llama3b_ppo_web/semantic_similarity_llama_refusal_full_search_prompt_2_ten_searches_web.json"),
        
        ("all_responses/llama3b_ppo_web/llama_refusal_full_search_prefill_1_loop_answer_web_new_stricter.json",
         "eval_results/llama3b_ppo_web/semantic_similarity_llama_refusal_full_search_prefill_1_loop_answer_web_new_stricter.json"),
        
        ("all_responses/llama3b_ppo_web/llama_refusal_full_search_prefill_2_loop_answer_web_new_stricter.json",
         "eval_results/llama3b_ppo_web/semantic_similarity_llama_refusal_full_search_prefill_2_loop_answer_web_new_stricter.json"),
        
        ("all_responses/llama3b_ppo_web/llama_refusal_full_search_prefill_3_loop_answer_web_new_stricter.json",
         "eval_results/llama3b_ppo_web/semantic_similarity_llama_refusal_full_search_prefill_3_loop_answer_web_new_stricter.json"),
        
        # LLAMA LOCAL
        ("all_responses/llama3b_ppo_local/llama_refusal_full_search.json",
         "eval_results/llama3b_ppo_local/semantic_similarity_llama_refusal_full_search.json"),
        
        ("all_responses/llama3b_ppo_local/llama_refusal_full_search_base.json",
         "eval_results/llama3b_ppo_local/semantic_similarity_llama_refusal_full_search_base.json"),
        
        ("all_responses/llama3b_ppo_local/llama_refusal_full_search_prefill_1_once.json",
         "eval_results/llama3b_ppo_local/semantic_similarity_llama_refusal_full_search_prefill_1_once.json"),
        
        ("all_responses/llama3b_ppo_local/llama_refusal_full_search_prefill_2_once.json",
         "eval_results/llama3b_ppo_local/semantic_similarity_llama_refusal_full_search_prefill_2_once.json"),
        
        ("all_responses/llama3b_ppo_local/llama_refusal_full_search_prefill_3_once.json",
         "eval_results/llama3b_ppo_local/semantic_similarity_llama_refusal_full_search_prefill_3_once.json"),
        
        ("all_responses/llama3b_ppo_local/llama_refusal_full_search_prompt_1.json",
         "eval_results/llama3b_ppo_local/semantic_similarity_llama_refusal_full_search_prompt_1.json"),
        
        ("all_responses/llama3b_ppo_local/llama_refusal_full_search_prompt_2.json",
         "eval_results/llama3b_ppo_local/semantic_similarity_llama_refusal_full_search_prompt_2.json"),
        
        ("all_responses/llama3b_ppo_local/llama_refusal_full_search_prompt_1_ten_searches.json",
         "eval_results/llama3b_ppo_local/semantic_similarity_llama_refusal_full_search_prompt_1_ten_searches.json"),
        
        ("all_responses/llama3b_ppo_local/llama_refusal_full_search_prompt_2_ten_searches.json",
         "eval_results/llama3b_ppo_local/semantic_similarity_llama_refusal_full_search_prompt_2_ten_searches.json"),
        
        ("all_responses/llama3b_ppo_local/llama_refusal_full_search_prefill_1_loop_answer_new_stricter.json",
         "eval_results/llama3b_ppo_local/semantic_similarity_llama_refusal_full_search_prefill_1_loop_answer_new_stricter.json"),
        
        ("all_responses/llama3b_ppo_local/llama_refusal_full_search_prefill_2_loop_answer_new_stricter.json",
         "eval_results/llama3b_ppo_local/semantic_similarity_llama_refusal_full_search_prefill_2_loop_answer_new_stricter.json"),
        
        ("all_responses/llama3b_ppo_local/llama_refusal_full_search_prefill_3_loop_answer_new_stricter.json",
         "eval_results/llama3b_ppo_local/semantic_similarity_llama_refusal_full_search_prefill_3_loop_answer_new_stricter.json"),
    ]
    
    # Load model once for all files
    print("="*80)
    print("BATCH SEMANTIC SIMILARITY PROCESSING")
    print("="*80)
    print(f"Files to process: {len(file_pairs)}")
    print("\nLoading sentence transformer model...")
    print("Using model: all-mpnet-base-v2 (high accuracy, state-of-the-art)")
    model = SentenceTransformer('all-mpnet-base-v2')
    print("✓ Model loaded successfully")
    print("="*80)
    
    # Process each file
    for i, (input_file, output_file) in enumerate(file_pairs, 1):
        print(f"\n{'='*80}")
        print(f"Processing file {i}/{len(file_pairs)}")
        print(f"{'='*80}")
        try:
            process_single_file(input_file, output_file, model)
        except Exception as e:
            print(f"ERROR processing {input_file}: {e}")
            continue
    
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

