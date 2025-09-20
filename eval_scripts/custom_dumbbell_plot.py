import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 10
})

metrics = ["Refusal", "Answer safety", "Search safety"]
x = np.arange(len(metrics))

# Data (0-100) pulled from your tables - reordered for layout
data = {
    "Qwen local": {
        "Base search":  np.array([38.5, 42.8, 11.2]),   # horizontal lines
        "Search":       np.array([92.0, 87.8, 70.8]),
        "Prefill1":     np.array([76.8, 73.3, 33.3])    # green (search attack)
    },
    "Qwen web": {
        "Base search":  np.array([38.5, 42.8, 11.2]),   # horizontal lines
        "Search":       np.array([91.0, 91.0, 64.2]),
        "Prefill1":     np.array([78.0, 74.8, 30.0])    # green (search attack)
    },
    "Llama local": {
        "Base search":  np.array([31.0, 40.0, 5.0]),    # horizontal lines
        "Search":       np.array([97.0, 96.2, 37.0]),
        "Prefill1":     np.array([74.0, 73.3, 14.0])    # green (search attack)
    },
    "Llama web": {
        "Base search":  np.array([31.0, 40.0, 5.0]),    # horizontal lines
        "Search":       np.array([97.2, 96.2, 35.8]),
        "Prefill1":     np.array([75.5, 75.0, 13.5])    # green (search attack)
    }
}

pal = {
    "Base search": "#666666",  # gray horizontal lines
    "Search": "#F39C12",     # orange baseline
    "Prefill1": "#2CA02C"    # green attack
}

fig, axs = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
axs = axs.ravel()

for ax, (title, vals) in zip(axs, data.items()):
    base_search = vals["Base search"]
    baseline = vals["Search"]
    g = vals["Prefill1"]

    # Plot base search as horizontal lines
    for i in range(len(x)):
        ax.axhline(y=base_search[i], xmin=(x[i]-0.4)/len(x), xmax=(x[i]+0.4)/len(x), 
                  color=pal["Base search"], linewidth=3, alpha=0.8, zorder=1, label="Base search" if i == 0 else "")

    # Plot baseline as open circles with thick vertical cap line for emphasis
    ax.scatter(x, baseline, s=140, color=pal["Search"], edgecolor="k", linewidth=1.0, zorder=3, label="Search (baseline)")
    # Plot search attack
    ax.scatter(x, g, s=120, color=pal["Prefill1"], edgecolor="k", linewidth=0.8, zorder=4, label="Search attack (Prefill1)")

    # Connect baseline to search attack with thin lines to form dumbbells
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [baseline[i], g[i]], color="#999999", linewidth=1.1, zorder=2)

    # Numeric labels (small) above markers
    for xi, base, b, gg in zip(x, base_search, baseline, g):
        ax.text(xi - 0.08, b + 2.0, f"{b:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(xi + 0.08, gg + 2.0, f"{gg:.1f}", ha="center", va="bottom", fontsize=9)
        # Add base search labels
        ax.text(xi, base - 3.0, f"{base:.1f}", ha="center", va="top", fontsize=9, color=pal["Base search"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(-2, 102)
    ax.set_xlim(-0.6, len(x)-0.2)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, color="#bfbfbf")
    ax.set_title(title, fontsize=13)
    # lighten frame
    for spine in ax.spines.values():
        spine.set_visible(False)

# Shared legend
handles = [
    plt.Line2D([0], [0], color=pal["Base search"], linewidth=3, label='Base search'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pal["Search"], markeredgecolor='k', markersize=10, label='Search (baseline)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pal["Prefill1"], markeredgecolor='k', markersize=10, label='Search attack (Prefill1)')
]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))

fig.suptitle("Safety metrics: Search (baseline) vs Search attack (green) vs Base search (gray lines)\n(0—100 scale)", fontsize=16, fontweight="bold")
plt.subplots_adjust(hspace=0.28, wspace=0.18, top=0.92, bottom=0.10)

# Save the figure
plt.savefig('/data/kebl6672/AGENTIC-RL/eval_scripts/custom_dumbbell_plot.png', 
            dpi=300, bbox_inches='tight')
plt.show()
print("Custom dumbbell plot saved to: /data/kebl6672/AGENTIC-RL/eval_scripts/custom_dumbbell_plot.png")