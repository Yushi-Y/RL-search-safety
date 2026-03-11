"""
Project qwen7b IT and Search-R1 (RL-ed) representations onto the search_query direction.

For each model (IT baseline + RL-trained Search-R1):
1. Load model
2. For each harmful prompt, format with search system prompt + prefilled <search>
3. Single forward pass, extract hidden state at last token (<search>)
4. Project onto search_query direction (attack vs natural, extracted from IT model)
5. Save mean projections per model

Hypothesis: RL training → larger projection onto search_query direction
(i.e., the model's <search> representation becomes more "attack-like" rather than "natural-like").
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuration ---
DIRECTION_FILE = Path("/VData/kebl6672/ARL/interp_results/directions/search_query_direction_qwen7b/search_query_direction.json")
HARMFUL_PATH = Path("/VData/kebl6672/ARL/refusal_datasets/harmful_combined.json")
OUTPUT_DIR = Path("/VData/kebl6672/ARL/interp_results/checkpoint_projections")

CKPT_BASE = Path("/VData/kebl6672/ARL/verl_checkpoints/search-r1-grpo-qwen2.5-7b-it-em/actor")

MODELS = {
    "Qwen2.5-7B-IT": "Qwen/Qwen2.5-7B-Instruct",
}

# Add all local 7B checkpoints
for step_dir in sorted(CKPT_BASE.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1])):
    step_num = int(step_dir.name.split("_")[-1])
    MODELS[f"Step {step_num}"] = str(step_dir)

# Search-R1 last (fully trained RL model)
MODELS["Search-R1 (RL)"] = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.2"

# Match extraction split: 600 sampled with seed 42 for direction extraction, rest held out
EXTRACTION_SEED = 42
EXTRACTION_N = 600

SEARCH_SYSTEM_PROMPT_TEMPLATE = """Answer the given question. \
You must conduct reasoning inside <think> and </think> every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. Question: {question}\n"""


def format_prompt(question, tokenizer):
    """Format with search system prompt + prefilled <search>."""
    question = question.strip()
    if question and question[-1] != '?':
        question += '?'
    content = SEARCH_SYSTEM_PROMPT_TEMPLATE.format(question=question)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt += "<search>" # attack
    return prompt


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load search_query direction
    if not DIRECTION_FILE.exists():
        print(f"ERROR: Direction file not found: {DIRECTION_FILE}")
        print("Run extract_search_query_d.py for qwen7b first.")
        return

    with open(DIRECTION_FILE) as f:
        directions = json.load(f)
    layer_keys = list(directions.keys())
    print(f"Direction layers: {layer_keys}")

    # Convert directions to tensors
    dir_tensors = {}
    for key in layer_keys:
        dir_tensors[key] = torch.tensor(directions[key], dtype=torch.float64)

    # Load held-out test prompts (complement of extraction train set)
    with open(HARMFUL_PATH) as f:
        all_harmful = [item["instruction"] for item in json.load(f)]
    random.seed(EXTRACTION_SEED)
    train_set = set(random.sample(all_harmful, EXTRACTION_N))
    prompts = [p for p in all_harmful if p not in train_set]
    print(f"Held-out test prompts: {len(prompts)} (from {len(all_harmful)} total, {len(train_set)} used for direction extraction)")

    print(f"Models to process: {list(MODELS.keys())}")

    # Process each model
    all_results = {}

    for ckpt_name, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {ckpt_name} ({model_path})")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        model.eval()

        # Extract layer indices from direction keys
        layer_indices = [int(k.replace("layer_", "")) for k in layer_keys]

        # Collect hidden states at <search> token
        reps = {k: [] for k in layer_keys}

        for inst in tqdm(prompts, desc=f"  {ckpt_name}"):
            prompt = format_prompt(inst, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, return_dict=True)

            for layer_idx, key in zip(layer_indices, layer_keys):
                # Last token is <search>
                reps[key].append(out.hidden_states[layer_idx][0, -1, :].cpu())


        torch.cuda.empty_cache()

        # Project onto search_query direction
        ckpt_results = {}
        for key in layer_keys:
            stacked = torch.stack(reps[key]).double()
            direction = dir_tensors[key]
            projs = stacked @ direction
            ckpt_results[key] = {
                "mean_proj": projs.mean().item(),
                "std_proj": projs.std().item(),
                "n_prompts": len(projs),
            }
            print(f"  {key}: mean_proj={projs.mean():.4f} ± {projs.std():.4f}")

        all_results[ckpt_name] = ckpt_results

        del model
        torch.cuda.empty_cache()

    # Save results
    output_file = OUTPUT_DIR / "qwen7b_all_checkpoints_search_query_projections.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_file}")

    # Plot: mean projection vs RL steps per layer
    fig, ax = plt.subplots(figsize=(14, 6))
    ckpt_names = list(all_results.keys())
    x = range(len(ckpt_names))

    for key in layer_keys:
        means = [all_results[c][key]["mean_proj"] for c in ckpt_names]
        sems = [all_results[c][key]["std_proj"] / (all_results[c][key]["n_prompts"] ** 0.5) for c in ckpt_names]
        ax.errorbar(x, means, yerr=sems, label=key, marker='o', capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(ckpt_names, rotation=45, ha='right')
    ax.set_ylabel("Mean projection onto search_query direction")
    ax.set_title("Qwen7B: <search> projection across training checkpoints")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_file = OUTPUT_DIR / "qwen7b_all_checkpoints_search_query_projection.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_file}")


if __name__ == "__main__":
    main()
