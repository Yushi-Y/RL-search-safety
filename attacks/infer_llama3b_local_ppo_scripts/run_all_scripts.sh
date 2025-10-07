#!/bin/bash

# Script to run all inference scripts in sequence
# Excludes: infer_search_base.py, infer_search_prompt_1.py, infer_search_prompt_2.py, 
#           infer_search_prompt_1_web.py, infer_search_prefill_ofcourse.py

set -e  # Exit on any error

echo "=========================================="
echo "Starting batch execution of inference scripts"
echo "=========================================="

# Change to the script directory
cd "$(dirname "$0")"

# List of scripts to run (in order)
scripts=(
    "infer_it_batch.py"
    "infer_search.py"
    "infer_search_prompt_sure.py"
    "infer_search_prefill_hereare.py"
    "infer_search_prefill_sure_1.py"
    "infer_search_prefill_sure_2.py"
    "infer_search_prefill_1_answer.py"
    "infer_search_prefill_1_beam_search.py"
    "infer_search_prefill_2_answer.py"
    "infer_search_prefill_2_beam_search.py"
    "infer_search_prefill_3_answer.py"
    "infer_search_prefill_4.py"
    "infer_search_prefill_5.py"
)

# Function to run a script with error handling
run_script() {
    local script=$1
    local script_num=$2
    local total_scripts=$3
    
    echo ""
    echo "=========================================="
    echo "Running script $script_num/$total_scripts: $script"
    echo "Start time: $(date)"
    echo "=========================================="
    
    if [ -f "$script" ]; then
        # Run the script and capture exit code
        if python "$script"; then
            echo ""
            echo "✅ Script $script completed successfully"
            echo "End time: $(date)"
        else
            echo ""
            echo "❌ Script $script failed with exit code $?"
            echo "End time: $(date)"
            echo ""
            echo "Do you want to continue with the next script? (y/n)"
            read -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                echo "Stopping execution due to user request"
                exit 1
            fi
        fi
    else
        echo "❌ Script $script not found, skipping..."
    fi
    
    echo ""
    echo "Waiting 5 seconds before next script..."
    sleep 5
}

# Run all scripts
total_scripts=${#scripts[@]}
for i in "${!scripts[@]}"; do
    script_num=$((i + 1))
    run_script "${scripts[$i]}" "$script_num" "$total_scripts"
done

echo ""
echo "=========================================="
echo "All scripts completed!"
echo "Final end time: $(date)"
echo "=========================================="