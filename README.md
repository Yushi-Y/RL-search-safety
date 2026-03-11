# Agentic RL for Search is Unsafe

Code for the paper *"Agentic Reinforcement Learning for Search is Unsafe"* (Yang et al., 2025). [[Paper]](https://arxiv.org/abs/2510.17431)

We train search-enabled LLMs (Llama and Qwen series) with RL (PPO/GRPO) following [Search-R1](https://github.com/PeterGriffinJin/Search-R1) (Jin et al., 2025), and evaluate their safety vulnerabilities under search-based jailbreak attacks.

## Repository Structure

```
ARL/
├── verl/                     # VERL RL training framework (modified from Search-R1)
├── setup/                    # Core components
│   ├── llm_agent/           # LLM agent implementation
│   └── retrieve/            # Retrieval server code
├── attacks/                  # Jailbreak attack inference scripts
│   ├── infer_engine.py      # Shared inference engine
│   └── run_infer.py         # Unified CLI for running attacks
├── eval_scripts/             # Evaluation (harm, refusal, search safety)
├── interp/                   # Interpretability analysis (directions, weights, steering)
├── figures/                  # Plotting scripts
├── analyse_search/           # Search behaviour analysis scripts
├── refusal_datasets/         # Harmful query datasets for attack evaluation
├── scripts/                  # Data processing utilities
├── train_ppo.sh              # PPO training script
└── train_grpo.sh             # GRPO training script
```

## Setup

### 1. Install Dependencies

We use two conda environments following [Search-R1](https://github.com/PeterGriffinJin/Search-R1):

**RL training environment:**
```bash
conda create -n rl python=3.9
conda activate rl
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install -e .
pip install flash-attn --no-build-isolation
pip install wandb
```

**Retriever environment:**
```bash
conda create -n retriever python=3.10
conda activate retriever
pip install transformers datasets pyserini uvicorn fastapi
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

### 2. Download Search Data

Download the pre-built FAISS index and Wikipedia corpus (see [Search-R1 data setup](https://github.com/PeterGriffinJin/Search-R1#data)):

```bash
save_path=search_data
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

### 3. Start Retrieval Server

```bash
conda activate retriever
bash retrieval_launch.sh
```

This starts the retrieval server at `http://127.0.0.1:8000/retrieve`.

### 4. Training Data

Download training data from [HuggingFace](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) and place in `data/nq_search/`.

## RL Training

### PPO
```bash
conda activate rl
bash train_ppo.sh
```

### GRPO
```bash
bash train_grpo.sh
```

Configure `BASE_MODEL`, `EXPERIMENT_NAME`, and `CUDA_VISIBLE_DEVICES` in the training scripts. Supported models include `meta-llama/Llama-3.2-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`.

## Jailbreak Attacks

All attack variants are run via a single CLI (`attacks/run_infer.py`). Attack types include prefill attacks (forcing `<search>` at the start of the response), prompt injections, and loop variants.

Examples:
```bash
conda activate retriever  # needs retrieval server running

# Prefill <search> once, then free generation
python attacks/run_infer.py --model qwen7b-it-ppo --prefill "<search>" --loop-mode once -o out.json

# Force 10 consecutive searches
python attacks/run_infer.py --model qwen7b-it-ppo --prefill "<search>" --loop-mode loop_prefill --max-searches 10 -o out.json

# Prompt injection
python attacks/run_infer.py --model qwen7b-it-ppo --prompt-injection prompt_1 -o out.json

# Web search backend
python attacks/run_infer.py --model llama3b-it-ppo --prefill "<search>" --loop-mode once --search-backend web -o out.json
```

Run `python attacks/run_infer.py --help` for all options. Responses are saved to the specified output file.

## Evaluation

Evaluation uses [Prometheus-eval](https://github.com/prometheus-eval/prometheus-eval) (LLM-based, 1-5 scale):

```bash
python eval_scripts/harm.py <responses.json>       # Harmfulness
python eval_scripts/refusal.py <responses.json>     # Refusal behaviour
python eval_scripts/search.py --input_file <responses.json> --output_file <results.json>  # Search safety
```

Results are saved to `eval_results/`.

## Citation

```bibtex
@misc{yang2025agenticreinforcementlearningsearch,
  title         = {Agentic Reinforcement Learning for Search is Unsafe},
  author        = {Yang, Yushi and Padarha, Shreyansh and Lee, Andrew and Mahdi, Adam},
  year          = {2025},
  eprint        = {2510.17431},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2510.17431}
}
```
