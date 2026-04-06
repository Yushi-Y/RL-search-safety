"""
Search safety under Search attack across RL training steps:
normal RL vs v1 vs v3.1 vs v3.2 vs v3.3 mitigation.
Scores: (original_score - 1) * 25, mapping 1-5 scale to 0-100.
"""

import matplotlib.pyplot as plt

# Normal RL (no mitigation)
steps_normal = [0, 25, 50, 75, 100]
normal_rl = [40.1, 20.6, 21.4, 21.8, 23.0]

# v1 mitigation (lambda=16, relu, mixed data)
steps_v1 = [0, 25, 50, 75, 100]
v1 = [40.1, 32.6, 33.6, 37.4, 38.6]

# v3.1 mitigation (lambda=9, relu, harmful data)
steps_v31 = [0, 25, 50, 75]
v31 = [40.1, 29.9, 27.8, 26.8]

# v3.2 mitigation (lambda=5, no relu, harmful data)
steps_v32 = [0, 25, 50, 75, 100]
v32 = [40.1, 31.3, 48.9, 46.5, 49.5]

# v3.3 mitigation (lambda=3, relu, forced search, harmful data)
steps_v33 = [0, 25, 50]
v33 = [40.1, 28.6, 25.7]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(steps_normal, normal_rl, 'o-', color='#ff7f0e', linewidth=2, markersize=6, label='No mitigation')
ax.plot(steps_v1, v1, 's-', color='#1f77b4', linewidth=2, markersize=6, label='v1 (λ=16, ReLU)')
ax.plot(steps_v31, v31, 'p-', color='#9467bd', linewidth=2, markersize=6, label='v3.1 (λ=9, ReLU)')
ax.plot(steps_v32, v32, 'D-', color='#2ca02c', linewidth=2, markersize=6, label='v3.2 (λ=5, no ReLU)')
ax.plot(steps_v33, v33, '^-', color='#d62728', linewidth=2, markersize=6, label='v3.3 (λ=3, ReLU, forced search)')

ax.set_xlabel('RL training steps', fontsize=12)
ax.set_ylabel('Average search safety (%)', fontsize=12)
ax.set_title('Search safety under prefill-once attack', fontsize=13)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_ylim(0, 100)
ax.legend(fontsize=9, loc='best')
ax.grid(False)

fig.tight_layout()
out_path = "/VData/kebl6672/ARL/figures/search_safety_v3.2_comparison.png"
fig.savefig(out_path, dpi=300)
plt.close()
print(f"Saved: {out_path}")
