#!/usr/bin/env python3
"""
Batch script to run search.py over all files in refusal_responses/qwen7b_ppo_local/
"""

import os
import subprocess
import glob
import json

def main():
    # Get the project root directory (parent of eval_scripts)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Directory containing the response files
    input_dir = os.path.join(project_root, "refusal_responses/qwen7b_ppo_local/")
    output_dir = os.path.join(project_root, "eval_results/qwen7b_ppo_local/")
    
    # Get all JSON files in the directory
    pattern = os.path.join(input_dir, "*.json")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} files to process:")
    for i, file in enumerate(files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    print(f"\nStarting batch processing...")
    
    # Process each file
    for i, input_file in enumerate(files, 1):
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"search_eval_{filename}")
        
        print(f"\n{'='*80}")
        print(f"Processing file {i}/{len(files)}: {filename}")
        print(f"Output: {output_file}")
        print(f"{'='*80}")
        
        # Create a temporary search.py with this specific file in eval_scripts directory
        temp_script = os.path.join(os.path.dirname(__file__), f"temp_search_{i}.py")
        
        # Read the original search.py
        script_path = os.path.join(project_root, "eval_scripts", "search.py")
        with open(script_path, "r") as f:
            content = f.read()
        
        # Replace the input and output files with absolute paths
        content = content.replace(
            'input_file = "refusal_responses/qwen7b_ppo_local/qwen_refusal_full_search.json"',
            f'input_file = "{os.path.abspath(input_file)}"'
        )
        content = content.replace(
            'output_file = "eval_results/qwen7b_ppo_local/qwen_results_search_full_search.json"',
            f'output_file = "{os.path.abspath(output_file)}"'
        )
        
        # Write temporary script
        with open(temp_script, "w") as f:
            f.write(content)
        
        try:
            # Run the temporary script from the eval_scripts directory
            temp_filename = os.path.basename(temp_script)
            result = subprocess.run(
                ["python", temp_filename],
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutes timeout per file
                cwd=os.path.join(project_root, "eval_scripts")  # Run from eval_scripts directory
            )
            
            if result.returncode == 0:
                # Check if the output file was actually created
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    print(f"✅ Successfully processed {filename} (saved {file_size} bytes)")
                else:
                    print(f"⚠️  Processing completed but output file not found: {output_file}")
            else:
                print(f"❌ Error processing {filename}")
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout processing {filename}")
        except Exception as e:
            print(f"❌ Exception processing {filename}: {e}")
        finally:
            # Clean up temporary script
            if os.path.exists(temp_script):
                os.remove(temp_script)
    
    print(f"\n{'='*80}")
    print("Batch processing complete!")
    print(f"Processed {len(files)} files")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()