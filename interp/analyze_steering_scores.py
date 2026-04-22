"""
Reanalyze scored steering results, filtering out empty (refused) responses.

For each layer × coefficient, reports:
  - total prompts
  - empty/refusal count
  - mean judge score (non-empty only)
  - score distribution (non-empty only)
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCORED_DIR = BASE_DIR / "interp_results/query_eval/llama_3b_steering"

FILES = {
    "search_search_attack":       SCORED_DIR / "steering_results_search_search_attack_scored.json",
    "search_query_search_attack": SCORED_DIR / "steering_results_search_query_search_attack_scored.json",
    "search_clean_search_attack": SCORED_DIR / "steering_results_search_clean_search_attack_scored.json",
}


def extract_query(item):
    if "query" in item:
        return item["query"]
    response = item.get("response", "")
    if "</search>" in response:
        return response.split("</search>")[0].strip()
    return ""


def analyze_file(path: Path, label: str):
    with open(path) as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"File: {label}")
    print(f"{'='*70}")
    print(f"{'Layer':<10} {'Coeff':>6}  {'Total':>6}  {'Empty':>6}  {'Judged':>6}  {'Mean':>6}  Dist(1-5)")
    print(f"{'-'*70}")

    for layer in sorted(data.keys(), key=lambda x: int(x.split("_")[1])):
        for coeff in sorted(data[layer].keys(), key=float):
            items = data[layer][coeff]
            total = len(items)
            empty = sum(1 for it in items if not extract_query(it).strip())
            scored = [it["score"] for it in items
                      if extract_query(it).strip() and isinstance(it.get("score"), (int, float))]
            n_judged = len(scored)
            mean = sum(scored) / n_judged if n_judged else float("nan")
            dist = {s: scored.count(s) for s in range(1, 6)}
            dist_str = " ".join(f"{dist.get(s,0)}" for s in range(1, 6))
            print(f"{layer:<10} {coeff:>6}  {total:>6}  {empty:>6}  {n_judged:>6}  {mean:>6.2f}  {dist_str}")


def main():
    for label, path in FILES.items():
        if not path.exists():
            print(f"Missing: {path}")
            continue
        analyze_file(path, label)
    print()


if __name__ == "__main__":
    main()
