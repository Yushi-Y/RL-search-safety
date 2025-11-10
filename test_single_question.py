import transformers
import torch
import requests
import json

# Model ID and device setup
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

curr_eos = [151645, 151643] # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# Initialize the tokenizer and model
print("Loading model and tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation="eager"
)
print("Model loaded successfully!")

# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
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
    
    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. Question: {question}\n"""

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    
    return prompt

def process_question(question_text):
    """Process a single question"""
    
    prompt = prepare_prompt(question_text)
    
    print(f'\n{"="*80}')
    print(f'Question: {question_text}')
    print(f'{"="*80}\n')
    
    # Initialize variables
    cnt = 0
    full_response = ""
    current_prompt = prompt
    search_information = []
    
    # Process the question with potential search iterations
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
            do_sample=False,
            use_cache=True
        )

        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            full_response += output_text
            print(output_text)
            break

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_response += output_text
        
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            print(f'\n[SEARCH] Query: "{tmp_query}"')
            search_results = search(tmp_query)
            search_information.append({
                "query": tmp_query,
                "results": search_results
            })
            print(f'[SEARCH] Got results (length: {len(search_results)} chars)')
        else:
            search_results = ''

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        current_prompt += search_text
        cnt += 1
        print(output_text)
    
    print(f'\n{"="*80}')
    print(f'COMPLETE - Total searches: {len(search_information)}')
    print(f'{"="*80}\n')
    
    return full_response, search_information


# The question to process
QUESTION = "Who won the 2018 presidential election in the country where the political party of Martín Ramírez Pech operates"

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Starting single question inference...")
    print("="*80 + "\n")
    
    response, search_info = process_question(QUESTION)
    
    print("\n" + "="*80)
    print("FINAL RESPONSE:")
    print("="*80)
    print(response)
    print("\n" + "="*80)
    print(f"Total searches performed: {len(search_info)}")
    print("="*80)
    
    # Save to file
    result = {
        "question": QUESTION,
        "response": response,
        "search_information": search_info,
        "num_searches": len(search_info)
    }
    
    output_file = "/data/kebl6672/ARL/single_question_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nResult saved to: {output_file}")

