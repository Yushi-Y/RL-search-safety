import transformers
import torch
import requests
import json
import gc
import re
import os

# Configuration variables
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
INPUT_FILE = os.path.join(PROJECT_ROOT, "refusal_datasets/harmful_full_original.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "all_responses/qwen32b_grpo_base.json")

# Model ID and device setup
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-32b-em-grpo-v0.3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

curr_eos = [151645, 151643]  # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"
)


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
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
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
    """Prepare prompt with the search system prompt (what the GRPO model was trained with)"""
    question = question_text.strip()
    if question[-1] != '?':
        question += '?'

    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. Question: {question}\n"""

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )

    return prompt


def process_single_question(question_text):
    """Process a single question — no prefill attack, model decides whether to search"""

    prompt = prepare_prompt(question_text)

    full_response = ""
    current_prompt = prompt
    search_information = []

    # No prefill — let the model generate freely
    while True:
        input_ids = tokenizer.encode(current_prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096 * 4,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            use_cache=True
        )

        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            full_response += output_text
            print(output_text[:200])
            break

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_response += output_text

        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            search_results = search(tmp_query)
            search_information.append({
                "query": tmp_query,
                "results": search_results
            })
        else:
            search_results = ''

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        current_prompt += search_text
        print(f"Search: {tmp_query}")

    torch.cuda.empty_cache()
    gc.collect()

    return full_response, search_information


def main():
    print(f"Loading questions from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)

    questions = [item.get("instruction", "") for item in questions_data if item.get("instruction", "")]
    print(f"Processing {len(questions)} questions...")

    results = []
    for i, question in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {question[:80]}...")

        try:
            response, search_info = process_single_question(question)
            result_entry = {
                "question": question,
                "response": response,
                "search_information": search_info,
                "question_index": i,
            }
            results.append(result_entry)
            print(f"  Search queries: {len(search_info)}")
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "question": question,
                "response": f"ERROR: {str(e)}",
                "search_information": [],
                "question_index": i,
            })

        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            print(f"  Saving progress... ({i+1}/{len(questions)})")
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done! {len(results)} responses saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
