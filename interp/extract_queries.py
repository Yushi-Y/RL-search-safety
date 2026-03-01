"""Extract first search query from steering results for analysis."""

import json
import os
import re

def extract_query(resp):
    """Response starts after prefilled <search>, so query is text before first </search>."""
    m = re.search(r'^(.*?)</search>', resp, re.DOTALL)
    return m.group(1).strip() if m else resp.strip()  # if no </search>, use full response


RESULTS_DIR = "interp_results/steering_results/qwen_7b_IT"

CONFIGS = [
    ("steering_results_search_search_attack.json", "search"),
    ("steering_results_search_query_search_attack.json", "search_query"),
    ("steering_results_search_clean_search_attack.json", "search_clean"),
    ("steering_results_refusal_search_attack.json", "refusal"),
]

LAYERS = ["layer_14", "layer_18", "layer_20", "layer_22", "layer_26"]
COEFFS = ["-2.0", "-1.0", "0.0", "1.0", "2.0"]

OUTPUT_DIR = "interp_results"
BASELINE_FILE = f"{OUTPUT_DIR}/queries_baseline.json"

# Extract baseline once (coeff=0.0, identical across all files/layers)
baseline_extracted = False
for fname, direction in CONFIGS:
    output_file = f"{OUTPUT_DIR}/queries_{direction}.json"
    if os.path.exists(output_file):
        print(f"Skipping {direction}: {output_file} already exists")
        continue

    with open(f"{RESULTS_DIR}/{fname}") as f:
        data = json.load(f)

    if not baseline_extracted:
        first_layer = LAYERS[0]
        baseline = [
            {"prompt": item["question"][:100], "query": extract_query(item["response"])}
            for item in data[first_layer]["0.0"]
        ]
        with open(BASELINE_FILE, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"Baseline extracted: {len(baseline)} queries -> {BASELINE_FILE}")
        baseline_extracted = True

    # Extract steered queries only (skip 0.0)
    out = {}
    for layer in LAYERS:
        out[layer] = {}
        for coeff in COEFFS:
            if coeff == "0.0":
                continue
            items = data[layer][coeff]
            out[layer][coeff] = [
                {
                    "prompt": item["question"][:100],
                    "query": extract_query(item["response"]),
                }
                for item in items
            ]

    with open(f"{OUTPUT_DIR}/queries_{direction}.json", "w") as f:
        json.dump(out, f, indent=2)

    # Print summary
    print(f"\n=== {direction} ===")
    baseline_queries = [x["query"] for x in baseline]
    for layer in LAYERS:
        for coeff in COEFFS:
            if coeff == "0.0":
                continue
            changed = sum(
                1 for b, s in zip(baseline_queries, [x["query"] for x in out[layer][coeff]])
                if b != s
            )
            print(f"  {layer} coeff={coeff}: {changed}/267 changed")
