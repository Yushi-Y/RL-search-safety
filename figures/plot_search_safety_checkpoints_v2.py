"""
Search safety under Search attack across RL training steps:
normal RL vs v1 vs v2 mitigation.
Scores: (original_score - 1) * 25, mapping 1-5 scale to 0-100.
"""

import matplotlib.pyplot as plt

# Normal RL (no mitigation)
steps_normal = [0, 25, 50, 75, 100]
normal_rl = [40.1, 19.6, 22.4, 27.8, 25.0]

# v1 mitigation (lambda=16, relu, mixed data)
steps_v1 = [0, 25, 50, 75, 100]
v1 = [40.1, 32.6, 33.6, 37.4, 38.6]

# v2 mitigation (lambda=8, no relu, mixed data)
steps_v2 = [0, 25, 50, 75, 100]
v2 = [40.1, 30.3, 29.2, 30.5, 33.0]


fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(steps_normal, normal_rl, 'o-', color='#ff7f0e', linewidth=2, markersize=6, label='No mitigation')
ax.plot(steps_v1, v1, 's-', color='#1f77b4', linewidth=2, markersize=6, label='v1 (λ=16, ReLU)')
ax.plot(steps_v2, v2, '^-', color='#d62728', linewidth=2, markersize=6, label='v2 (λ=8, no ReLU)')

ax.set_xlabel('RL training steps', fontsize=12)
ax.set_ylabel('Average search safety (%)', fontsize=12)
ax.set_title('Search safety under prefill-once attack', fontsize=13)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_ylim(0, 60)
ax.legend(fontsize=10, loc='best')

fig.tight_layout()
out_path = "/VData/kebl6672/ARL/figures/search_safety_v2.png"
fig.savefig(out_path, dpi=300)
plt.close()
print(f"Saved: {out_path}")
