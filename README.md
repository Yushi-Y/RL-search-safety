# RL-search-safety

Code for paper "Agentic RL for search is unsafe"

This repository contains code for training RL-based search agents and evaluating their safety vulnerabilities.

## Overview

This project implements reinforcement learning (RL) training for search-enabled LLMs (Qwen and Llama-series) using PPO and GRPO algorithms. The training pipeline follows *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning* (Jin et al., 2025) [Pape](https://arxiv.org/abs/2503.09516). 

It includes evaluation frameworks for assessing search quality, harmfulness, and refusal behaviours, as well as jailbreak attack scripts (Search and Multi-search attacks) for safety testing.

## Structure

```
RL-search-safety/
├── setup/                    # Core components
│   ├── llm_agent/           # LLM agent implementation
│   └── retrieve/            # Retrieval/search server code
├── verl/                     # VERL RL training framework
├── attacks/                  # Jailbreak attack scripts
├── eval_scripts/             # Evaluation code
├── scripts/                  # Data processing and utilities
├── refusal_datasets/         # Harmful query datasets
└── train_*.sh               # RL training scripts
```

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `vllm`, `accelerate`, `ray`, `wandb`, `prometheus_eval`

## RL Training

### PPO Training

Train a search agent using Proximal Policy Optimization:

```bash
bash train_ppo.sh
```

Configure in `train_ppo.sh`:
- `BASE_MODEL`: Base model (e.g., `meta-llama/Llama-3.2-3B`, `Qwen/Qwen2.5-7B`)
- `EXPERIMENT_NAME`: Experiment identifier
- `DATA_DIR`: Training data directory
- `CUDA_VISIBLE_DEVICES`: GPU devices

### GRPO Training

Train using Group Relative Policy Optimization:

```bash
bash train_grpo.sh
```


### Training Data

Training data should be in parquet format at `data/nq_search/train.parquet`. The data format includes search queries, retrieved results, and agent responses.

## Jailbreak Attacks

The `attacks/` directory contains scripts for testing model safety with harmful queries:

- **Inference scripts**: Generate responses to harmful queries using trained models
- **Attack variants**: Different prompting strategies (prefill, loop, prompt-based)
- **Model-specific**: Scripts for different model sizes (3B, 7B, 14B)


## Evaluation

### Search Quality Evaluation

Evaluate search quality using Prometheus evaluator:

```bash
python eval_scripts/search.py --input_file <responses.json> --output_file <results.json>
```

### Harmfulness Evaluation

Assess if responses contain harmful content:

```bash
python eval_scripts/harm.py <input_file>
```

### Refusal Evaluation

Check if models refuse harmful queries:

```bash
python eval_scripts/refusal.py <input_file>
```

### Semantic Similarity

Evaluate semantic similarity between queries and responses:

```bash
python eval_scripts/semantic_similarity_llama.py <input_file>
python eval_scripts/semantic_similarity_qwen.py <input_file>
```

### Batch Evaluation

Run batch evaluation on multiple files:

```bash
python eval_scripts/batch_search_eval_qwen_local.py
python eval_scripts/batch_ss_eval_llama_local.py
```

## Data

- **Training data**: `data/nq_search/` - Natural Questions dataset for RL training
- **Harmful datasets**: `refusal_datasets/` - Datasets of harmful queries for safety testing
  - `harmful_full_questions.json`
  - `harmful_full_imperative.json`
  - `harmful_full_original.json`

## Outputs

- **Model responses**: `all_responses/`, `refusal_responses/` - Generated responses from attacks
- **Evaluation results**: `eval_results/` - Evaluation outputs 
- **Figures**: `figures/` - Analysis plots and visualizations 

## Analysis

The `analyse_search/` directory contains scripts for analyzing attack results, safety scores, and search behavior patterns.


## Citation

If you find this code helpful, please cite the paper:

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
