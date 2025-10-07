#!/bin/bash

# Batch 1: Run first 7 scripts
echo "Starting Batch 1 - Running 7 scripts..."

# Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate eval

cd /data/kebl6672/ARL

echo "Running infer.py..."
python attacks/infer_v2_local/infer.py
echo "infer.py completed!"

echo "Running infer_prefill_1_loop_answer.py..."
python attacks/infer_v2_local/infer_prefill_1_loop_answer.py
echo "infer_prefill_1_loop_answer.py completed!"

echo "Running infer_prefill_1_once.py..."
python attacks/infer_v2_local/infer_prefill_1_once.py
echo "infer_prefill_1_once.py completed!"

echo "Running infer_prefill_2_loop_answer.py..."
python attacks/infer_v2_local/infer_prefill_2_loop_answer.py
echo "infer_prefill_2_loop_answer.py completed!"

echo "Running infer_prefill_2_once.py..."
python attacks/infer_v2_local/infer_prefill_2_once.py
echo "infer_prefill_2_once.py completed!"

echo "Running infer_prefill_3_loop_answer.py..."
python attacks/infer_v2_local/infer_prefill_3_loop_answer.py
echo "infer_prefill_3_loop_answer.py completed!"

echo "Batch 1 completed!"