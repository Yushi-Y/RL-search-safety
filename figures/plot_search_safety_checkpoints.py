"""
Search safety under Search attack across RL training steps:
normal RL vs v1 mitigation. Two subfigures: search safety and QA EM.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

steps = [0, 25, 50, 75, 100]
# Search safety
no_mitigation = [40.1, 20.6, 21.4, 21.8, 23.0]
v1_mitigation = [40, 32, 33, 37, 38]

# QA EM scores
em_no_mitigation = [0.285, 0.330, 0.347, 0.359, 0.378]
em_v1_mitigation = [0.285, 0.304, 0.331, 0.341, 0.356]

color_no_mit = '#ff7f0e'
color_v1_mit = '#1f77b4'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

# Left: search safety
ax1.plot(steps, no_mitigation, 'o-', color=color_no_mit, linewidth=2, markersize=6, label='No mitigation')
ax1.plot(steps, v1_mitigation, 's-', color=color_v1_mit, linewidth=2, markersize=6, label='v1 mitigation')
ax1.set_xlabel('RL training steps', fontsize=12)
ax1.set_ylabel('Average search safety', fontsize=12)
ax1.set_title('Search safety under Search attack', fontsize=13)
ax1.set_xticks(steps)
ax1.set_ylim(0, 60)
ax1.legend(fontsize=10)

# Right: QA EM
ax2.plot(steps, em_no_mitigation, 'o-', color=color_no_mit, linewidth=2, markersize=6, label='No mitigation')
ax2.plot(steps, em_v1_mitigation, 's-', color=color_v1_mit, linewidth=2, markersize=6, label='v1 mitigation')
ax2.set_xlabel('RL training steps', fontsize=12)
ax2.set_ylabel('QA test set EM score', fontsize=12)
ax2.set_title('Benign QA performance', fontsize=13)
ax2.set_xticks(steps)
ax2.legend(fontsize=10)

fig.tight_layout()
out_path = "/VData/kebl6672/ARL/figures/search_safety_checkpoints.png"
fig.savefig(out_path, dpi=300)
plt.close()
print(f"Saved: {out_path}")
