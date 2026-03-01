"""
Extract search harmfulness direction from IT models.

Approach:
1. For each prompt (harmful and benign), format with search system prompt + append <search>
2. Forward pass, grab hidden state at last token (= the <search> token)
3. Direction d = mean(harmful) - mean(benign), normalized

NOTE: Uses the same system prompt as the actual attack scenario so the extraction
context matches where steering will be applied.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Configuration
MODELS = {
    "llama3b": {
        "model_path": "meta-llama/Llama-3.2-3B-Instruct",
        "layers": [14, 18, 20, 22, 26, 28],  # 28 layers (0-27); 18,20,22 = steering targets
    },
    "qwen3b": {
        "model_path": "Qwen/Qwen2.5-3B-Instruct",
        "layers": [10, 14, 18, 22, 26, 30],  # 36 layers (0-35)
    },
    "qwen7b": {
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "layers": [14, 18, 20, 22, 26, 28],  # 28 layers (0-27); 22 = steering target
    },
    "qwen14b": {
        "model_path": "Qwen/Qwen2.5-14B-Instruct",
        "layers": [24, 32, 36, 40, 44, 48],  # 48 layers (0-47); 36 = steering target
    },
    "qwen32b": {
        "model_path": "Qwen/Qwen2.5-32B-Instruct",
        "layers": [32, 42, 50, 52, 60, 64],  # 64 layers (0-63); 50 = steering target
    },
}

HARMFUL_PATH = Path("/VData/kebl6672/ARL/refusal_datasets/harmful_combined.json")
HARMLESS_PATH = Path("/VData/kebl6672/ARL/harmless_datasets/harmless_val.json")

MAX_HARMFUL = 600   # 70% train split (871 total → 600 train / 271 held-out)
MAX_HARMLESS = 600
RANDOM_SEED = 42


def prepare_prompt(instruction, tokenizer, use_sys_prompt=True):
    """Format instruction with optional search system prompt + <search> token."""
    question = instruction.strip()
    if question and question[-1] != '?':
        question += '?'

    if use_sys_prompt:
        content = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. Question: {question}\n"""
    else:
        content = question

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        prompt = content

    return prompt + "<search>"


def get_hidden_state(model, tokenizer, instruction, layer_idx, use_sys_prompt=True):
    """Get hidden state at the <search> token (last token) for one instruction."""
    prompt = prepare_prompt(instruction, tokenizer, use_sys_prompt=use_sys_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # Last token = <search> token (appended at end of prompt)
    return outputs.hidden_states[layer_idx][0, -1, :].cpu() # [1, seq_len, d]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sys-prompt", action="store_true", help="Extract clean variant without search system prompt")
    args = parser.parse_args()
    use_sys_prompt = not args.no_sys_prompt

    random.seed(RANDOM_SEED)

    # Load instructions
    with open(HARMFUL_PATH) as f:
        harmful = [item['instruction'] for item in json.load(f)]
    with open(HARMLESS_PATH) as f:
        harmless = [item['instruction'] for item in json.load(f)]
    if MAX_HARMFUL and len(harmful) > MAX_HARMFUL:
        harmful = random.sample(harmful, MAX_HARMFUL)
    if MAX_HARMLESS and len(harmless) > MAX_HARMLESS:
        harmless = random.sample(harmless, MAX_HARMLESS)
    print(f"Harmful: {len(harmful)}, Harmless: {len(harmless)}")

    for model_name, model_cfg in MODELS.items():
        model_path = model_cfg["model_path"]
        layer_indices = model_cfg["layers"]
        direction_type = "search" if use_sys_prompt else "search_clean"
        direction_name = f"{direction_type}_direction"
        output_dir = Path(f"/VData/kebl6672/ARL/interp_results/directions/{direction_name}_{model_name}")

        if (output_dir / f"{direction_name}.json").exists():
            print(f"\nSkipping {model_name}: {output_dir / f'{direction_name}.json'} already exists")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing {model_name}: {model_path}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
        )
        model.eval()

        all_directions = {}
        all_stats = {}

        for layer_idx in layer_indices:
            print(f"\nLayer {layer_idx}:")

            harmful_reps = torch.stack([
                get_hidden_state(model, tokenizer, inst, layer_idx, use_sys_prompt=use_sys_prompt)
                for inst in tqdm(harmful, desc="  Harmful")
            ]).double()  # float64 for numerical stability 
            benign_reps = torch.stack([
                get_hidden_state(model, tokenizer, inst, layer_idx, use_sys_prompt=use_sys_prompt)
                for inst in tqdm(harmless, desc="  Benign")
            ]).double()  # float64 for numerical stability 

            direction_raw = harmful_reps.mean(0) - benign_reps.mean(0)
            direction = direction_raw / direction_raw.norm()

            harmful_projs = harmful_reps @ direction
            benign_projs = benign_reps @ direction
            separation = harmful_projs.mean().item() - benign_projs.mean().item()

            print(f"  Separation: {separation:.4f}")
            print(f"  Harmful proj: {harmful_projs.mean():.4f} +/- {harmful_projs.std():.4f}")
            print(f"  Benign proj:  {benign_projs.mean():.4f} +/- {benign_projs.std():.4f}")

            key = f"layer_{layer_idx}"
            all_directions[key] = direction.float().numpy().tolist()
            all_stats[key] = {
                "separation": separation,
                "direction_norm": direction_raw.norm().item(),
                "harmful_proj_mean": harmful_projs.mean().item(),
                "harmful_proj_std": harmful_projs.std().item(),
                "benign_proj_mean": benign_projs.mean().item(),
                "benign_proj_std": benign_projs.std().item(),
            }

        del model
        torch.cuda.empty_cache()

        # Save
        with open(output_dir / f"{direction_name}.json", "w") as f:
            json.dump(all_directions, f, indent=2)
        with open(output_dir / f"{direction_type}_stats.json", "w") as f:
            json.dump(all_stats, f, indent=2)

        # Plot separation by layer
        fig, ax = plt.subplots(figsize=(10, 6))
        seps = [all_stats[f"layer_{l}"]["separation"] for l in layer_indices]
        ax.bar(range(len(layer_indices)), seps, color='#2ecc71', alpha=0.8)
        ax.set_xticks(range(len(layer_indices)))
        ax.set_xticklabels([f'Layer {l}' for l in layer_indices])
        ax.set_ylabel('Separation (harmful - benign)')
        ax.set_title(f'{model_name}: Search Direction Separation by Layer')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / "separation_by_layer.png", dpi=300)
        plt.close()

        best = max(layer_indices, key=lambda l: all_stats[f"layer_{l}"]["separation"])
        print(f"\nBest layer: {best} (separation={all_stats[f'layer_{best}']['separation']:.4f})")
        print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
