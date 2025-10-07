#!/bin/bash

# Batch 2: Run remaining 7 scripts
echo "Starting Batch 2 - Running 7 scripts..."

# Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate eval

cd /data/kebl6672/ARL

echo "Running infer_prefill_sure_1.py..."
python attacks/infer_v2_local/infer_prefill_sure_1.py
echo "infer_prefill_sure_1.py completed!"

echo "Running infer_prefill_sure_2.py..."
python attacks/infer_v2_local/infer_prefill_sure_2.py
echo "infer_prefill_sure_2.py completed!"

echo "Running infer_prompt_1.py..."
python attacks/infer_v2_local/infer_prompt_1.py
echo "infer_prompt_1.py completed!"

echo "Running infer_prompt_1_ten_times.py..."
python attacks/infer_v2_local/infer_prompt_1_ten_times.py
echo "infer_prompt_1_ten_times.py completed!"

echo "Running infer_prompt_2.py..."
python attacks/infer_v2_local/infer_prompt_2.py
echo "infer_prompt_2.py completed!"

echo "Running infer_prompt_2_ten_times.py..."
python attacks/infer_v2_local/infer_prompt_2_ten_times.py
echo "infer_prompt_2_ten_times.py completed!"

echo "Batch 2 completed!"