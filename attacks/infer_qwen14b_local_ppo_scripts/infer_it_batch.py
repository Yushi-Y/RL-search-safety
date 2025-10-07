import transformers
import torch
import random
from datasets import load_dataset
import json
import time
import gc

# Configuration variables
INPUT_FILE = "refusal_datasets/arditi_harmful_full.json"
OUTPUT_FILE = "refusal_responses/qwen14b_refusal_full_it.json"
BATCH_SIZE = 1
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

# Set padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation="eager"  # Disable flash attention
)

# Define the custom stopping criterion for </answer>
class StopOnAnswer(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

# Initialize the stopping criteria
target_sequences = ["</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnAnswer(target_sequences, tokenizer)])

def prepare_prompt(question_text):
    """Prepare a single prompt for a question"""
    question = question_text.strip()
    if question[-1] != '?':
        question += '?'
    
    # Prepare the message - simplified version without search functionality
    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first. \
After reasoning, provide the answer inside <answer> and </answer>. \
Question: {question}\n"""

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    
    return prompt

def process_batch(questions, questions_data, output_file, batch_size=BATCH_SIZE):
    """Process a batch of questions using the IT model"""
    
    # Prepare all prompts
    prompts = [prepare_prompt(q) for q in questions]
    
    all_results = []
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_questions = questions[i:i+batch_size]
        
        print(f'\n\n################# [Processing Batch {i//batch_size + 1}] ##################\n\n')
        print(f"Processing {len(batch_prompts)} questions in batch...")
        
        # Tokenize the batch
        encoded = tokenizer(
            batch_prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=4096*2
        ).to(DEVICE)
        
        # Generate for the batch
        with torch.no_grad():
            outputs = model.generate(
                encoded.input_ids,
                attention_mask=encoded.attention_mask,
                max_new_tokens=4096*4,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # Greedy decoding (deterministic)
                use_cache=True  # Enable KV caching for faster generation
            )
        
        # Clear GPU memory after each batch
        torch.cuda.empty_cache()
        gc.collect()
        
        # Decode results for this batch
        batch_results = []
        for j, (input_ids, output_ids, question) in enumerate(zip(encoded.input_ids, outputs, batch_questions)):
            # Extract only the generated tokens
            generated_tokens = output_ids[len(input_ids):]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"Question {i+j+1}: {question[:100]}...")
            print(f"Response: {output_text[:200]}...")
            print("-" * 50)
            
            batch_results.append(output_text)
            
            # Add this result to all results
            all_results.append(output_text)
            
            # Save progress every 10 questions
            if len(all_results) % 10 == 0 or len(all_results) == len(questions):
                # Create result entries for processed items so far
                results = []
                for idx, (item, response) in enumerate(zip(questions_data[:len(all_results)], all_results)):
                    result_entry = {
                        "question": item.get("instruction", "") or item.get("question", ""),
                        "response": response,
                        "question_index": idx,
                    }
                    results.append(result_entry)
                
                print(f"\nSaving progress... ({len(all_results)}/{len(questions)} questions)")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
    
    return all_results

def main():
    # Load the questions from the JSON file
    print(f"Loading questions from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    print(f"Found {len(questions_data)} questions to process")
    
    # Extract questions
    questions = [item.get("instruction", "") or item.get("question", "") for item in questions_data if item.get("instruction", "") or item.get("question", "")]
    
    print(f"Processing {len(questions)} valid questions in batches...")
    
    try:
        # Process all questions in batches with per-batch saving
        all_responses = process_batch(questions, questions_data, OUTPUT_FILE, batch_size=BATCH_SIZE)
        
        print(f"Processing complete! Results saved to {OUTPUT_FILE}")
        print(f"Successfully processed {len(all_responses)} questions")
        
    except Exception as e:
        print(f"Error during batch processing: {e}")
        print("Falling back to individual processing...")
        
        # Fallback to individual processing
        results = []
        for i, item in enumerate(questions_data):
            question = item.get("instruction", "") or item.get("question", "")
            if not question:
                continue
                
            print(f"Processing question {i+1}/{len(questions_data)}")
            
            try:
                # Process single question
                temp_data = [{"question": question}]
                responses = process_batch([question], temp_data, OUTPUT_FILE, batch_size=1)
                response = responses[0] if responses else "ERROR: No response generated"
                
                result_entry = {
                    "question": question,
                    "response": response,
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