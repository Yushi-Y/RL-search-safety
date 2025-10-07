import transformers
import torch
import random
from datasets import load_dataset
import requests
import json
import time
import gc

# Configuration variables
INPUT_FILE = "refusal_datasets/arditi_harmful_full.json"
OUTPUT_FILE = "refusal_responses/qwen_refusal_full_search_prefill_ofcourse.json"

# Model ID and device setup
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

curr_eos = [151645, 151643] # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
# model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation="eager"  # Disable flash attention
)

# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])

# Initialize the stopping criteria
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

def prepare_prompt(question_text):
    """Prepare a single prompt for a question"""
    question = question_text.strip()
    if question[-1] != '?':
        question += '?'
    
    # Prepare the message - exactly the same as infer_search.py
    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. Question: {question}\n"""

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    
    return prompt

def process_single_question(question_text):
    """Process a single question using the same logic as infer_search.py"""
    
    prompt = prepare_prompt(question_text)
    
    print(f'\n\n################# [Start Reasoning + Searching] ##################\n\n')
    print(f"Question: {question_text}")
    
    # Initialize variables exactly like infer_search.py
    cnt = 0
    full_response = ""
    current_prompt = prompt
    search_information = []  # Store all search queries and results
    
    # Prefill with "Of course, " only once at the beginning
    current_prompt += "Of course, "
    
    # Generate freely from here on
    while True:
        input_ids = tokenizer.encode(current_prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096*4,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False, # Greedy decoding (temperature=0.0)
            use_cache=True  # Enable KV caching for faster generation
        )

        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # Add the prefilled text to the response since it's not in generated_tokens
            full_response += "Of course, " + output_text
            print("Of course, " + output_text)
            break

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # Add the prefilled text to the response since it's not in generated_tokens
        full_response += "Of course, " + output_text
        
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            search_results = search(tmp_query)
            # Store the search query and results
            search_information.append({
                "query": tmp_query,
                "results": search_results
            })
        else:
            search_results = ''

        search_text = curr_search_template.format(output_text="Of course, " + output_text, search_results=search_results)
        current_prompt += search_text
        print(f"Search: {search_text}")
    
    # Clear GPU memory after processing
    torch.cuda.empty_cache()
    gc.collect()
    
    return full_response, search_information

def process_questions_sequential(questions, questions_data, output_file, save_interval=10):
    """Process questions sequentially with periodic saving"""
    
    results = []
    
    # Process questions individually (search requires sequential processing)
    for i, question in enumerate(questions):
        print(f'\n\n################# [Processing Question {i+1}/{len(questions)}] ##################\n\n')
        
        try:
            # Process the question using the search logic
            response, search_info = process_single_question(question)
            
            # Create result entry
            result_entry = {
                "question": question,
                "response": response,
                "search_information": search_info,
                "question_index": i,
            }
            results.append(result_entry)
            
            print(f"Question {i+1}: {question[:100]}...")
            print(f"Response: {response[:200]}...")
            print(f"Search queries: {len(search_info)}")
            print("-" * 50)
            
            # Save progress every save_interval questions
            if (i + 1) % save_interval == 0 or (i + 1) == len(questions):
                print(f"\nSaving progress... ({i+1}/{len(questions)} questions)")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Small delay to prevent overwhelming the search service
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            result_entry = {
                "question": question,
                "response": f"ERROR: {str(e)}",
                "search_information": [],
                "question_index": i,
            }
            results.append(result_entry)
    
    return [r["response"] for r in results]

def main():
    # Load the questions from the JSON file
    print(f"Loading questions from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    print(f"Found {len(questions_data)} questions to process")
    
    # Extract questions
    questions = [item.get("instruction", "") for item in questions_data if item.get("instruction", "")]
    
    print(f"Processing {len(questions)} valid questions sequentially...")
    
    try:
        # Process all questions sequentially with periodic saving
        all_responses = process_questions_sequential(questions, questions_data, OUTPUT_FILE, save_interval=10)
        
        print(f"Processing complete! Results saved to {OUTPUT_FILE}")
        print(f"Successfully processed {len(all_responses)} questions")
        
    except Exception as e:
        print(f"Error during sequential processing: {e}")
        print("Falling back to individual processing...")
        
        # Fallback to individual processing
        results = []
        for i, item in enumerate(questions_data):
            # Try both field names to be safe
            question = item.get("instruction", "") or item.get("question", "")
            if not question:
                continue
                
            print(f"Processing question {i+1}/{len(questions_data)}")
            
            try:
                # Process single question
                response, search_info = process_single_question(question)
                
                result_entry = {
                    "question": question,
                    "response": response,
                    "search_information": search_info,
                    "question_index": i,
                }
                results.append(result_entry)
                
                # Save progress every 10 questions
                if (i + 1) % 10 == 0:
                    print(f"Saving progress... ({i+1}/{len(questions_data)})")
                    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                
            except Exception as individual_error:
                print(f"Error processing question {i+1}: {individual_error}")
                result_entry = {
                    "question": question,
                    "response": f"ERROR: {str(individual_error)}",
                    "search_information": [],
                    "question_index": i,
                }
                results.append(result_entry)
        
        # Save final results from fallback
        print(f"\nSaving final results to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Fallback processing complete! Results saved to {OUTPUT_FILE}")
        print(f"Successfully processed {len(results)} questions")

if __name__ == "__main__":
    main() 