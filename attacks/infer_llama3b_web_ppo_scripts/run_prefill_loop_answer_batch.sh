#!/bin/bash

# Run prefill loop_answer scripts in sequence
echo "Starting Llama3b Web Prefill Loop Answer Sequence..."

# Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate eval

cd /data/kebl6672/ARL

echo "Running infer_search_prefill_1_loop_answer_web.py..."
python attacks/infer_llama3b_web_ppo_scripts/infer_search_prefill_1_loop_answer_web.py
echo "infer_search_prefill_1_loop_answer_web.py completed!"

echo "Running infer_search_prefill_2_loop_answer_web.py..."
python attacks/infer_llama3b_web_ppo_scripts/infer_search_prefill_2_loop_answer_web.py
echo "infer_search_prefill_2_loop_answer_web.py completed!"

echo "Running infer_search_prefill_3_loop_answer_web.py..."
python attacks/infer_llama3b_web_ppo_scripts/infer_search_prefill_3_loop_answer_web.py
echo "infer_search_prefill_3_loop_answer_web.py completed!"

echo "Running infer_search_prefill_3_loop_answer.py (local)..."
python attacks/infer_llama3b_local_ppo_scripts/infer_search_prefill_3_loop_answer.py
echo "infer_search_prefill_3_loop_answer.py (local) completed!"

echo "All prefill loop_answer scripts completed!"