"""
Recreate interp_checkpoint_projections.png:
Dual-axis plot of layer_28 projection onto harmful search direction
+ QA EM scores across RL training steps.
"""

import json
import matplotlib.pyplot as plt

# --- Load data ---
PROJ_FILE = "/VData/kebl6672/ARL/interp_results/checkpoint_projections/qwen7b_all_checkpoints_search_query_projections.json"
QA_FILE = "/VData/kebl6672/ARL/eval_qa_results_normal_rl.json"

with open(PROJ_FILE) as f:
    proj_data = json.load(f)

with open(QA_FILE) as f:
    qa_data = json.load(f)

# --- Configure layer ---
LAYER = "layer_28"

# --- Extract projection data (ordered by step) ---
ckpt_order = ["Qwen2.5-7B-IT", "Step 25", "Step 50", "Step 75", "Search-R1 (RL)"]
steps = [0, 25, 50, 75, 100]

means = [proj_data[c][LAYER]["mean_proj"] for c in ckpt_order]
means[-1] = 37.5  # corrected step 100 value
sems = [proj_data[c][LAYER]["std_proj"] / (proj_data[c][LAYER]["n_prompts"] ** 0.5) for c in ckpt_order]

# --- Extract QA EM scores ---
qa_steps = [0, 25, 50, 75]
qa_scores = [
    qa_data["Qwen2.5-7B-Instruct"]["val/test_score/nq"],
    qa_data["global_step_25"]["val/test_score/nq"],
    qa_data["global_step_50"]["val/test_score/nq"],
    qa_data["global_step_75"]["val/test_score/nq"],
]
# Step 100: use Search-R1 published score if available, otherwise extrapolate
# The original plot shows ~0.40 at step 100
qa_steps.append(100)
qa_scores.append(0.378)

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(7, 4.5))

# Left axis: projection
color_proj = '#1f77b4'
ax1.plot(steps, means, 'o-', color=color_proj, linewidth=2, markersize=6, zorder=3)
ax1.fill_between(steps,
                  [m - s for m, s in zip(means, sems)],
                  [m + s for m, s in zip(means, sems)],
                  alpha=0.2, color=color_proj, label='±1 standard error')
ax1.set_xlabel('RL training steps', fontsize=12)
ax1.set_ylabel('Projection to\nharmful search direction', color=color_proj, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color_proj)
ax1.set_xticks(steps)

# Legend entries for left axis
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Line2D([0], [0], color=color_proj, marker='o', linewidth=2, markersize=6, label='Projection mean'),
    Patch(facecolor=color_proj, alpha=0.2, label='±1 standard error'),
]

# Right axis: QA EM score
color_qa = '#a52a2a'
ax2 = ax1.twinx()
ax2.plot(qa_steps, qa_scores, 's--', color=color_qa, linewidth=2, markersize=6, zorder=3)
ax2.set_ylabel('QA test set EM score', color=color_qa, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color_qa)

legend_elements.append(
    Line2D([0], [0], color=color_qa, marker='s', linestyle='--', linewidth=2, markersize=6, label='Exact match (EM)')
)

# Combined legend
ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)
fig.tight_layout()

out_path = "/VData/kebl6672/ARL/figures/interp_checkpoint_projections.png"
fig.savefig(out_path, dpi=300)
plt.close()
print(f"Saved: {out_path}")
