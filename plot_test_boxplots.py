"""
Comparison of ASIA results against literature RMSE values, across all
nonlinear system identification benchmarks with a multi-value leaderboard.

Usage:
1. Fill in ASIA_RESULTS below with your obtained value(s) once training is done
   (replace "TBD" with a float, or a list of floats for multi-condition benchmarks).
2. Run: python benchmark_comparison_plot.py
3. Output figure is saved to output/benchmark_comparison_all.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

tex_fonts = {
    "text.usetex": False,
    "font.family": "serif",
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}
plt.rcParams.update(tex_fonts)


os.makedirs('output', exist_ok=True)

# ============================================================
# RAW LITERATURE DATA
# Single-condition benchmarks: a flat list of scalars (one per paper/algorithm).
# Multi-condition benchmarks: a list of rows, each row = list of scalars
# (one per output/test condition), missing entries marked as None.
# ============================================================

WH = [0.241, 0.279, 0.3, 0.3, 0.42, 0.49, 1.52, 1.54, 1.75, 1.77,
      4.07, 4.88, 11.8, 12.8, 17.3, 30.5, 43.4, 87.2]

EMPS = [2.64, 2.65, 3.68, 4.61, 4.61, 4.73, 5.01, 6.65, 11, 17.6,
        42, 78, 80, 85.2, 86.2, 95.9]

Silverbox = [
    [0.289, 0.334, 0.257], [0.36, 1.4, 0.32], [0.402, 0.881, 0.368],
    [0.418, 0.772, 0.373], [0.505, 1.41, 0.455], [1.04, 5.87, 1.27],
    [1.05, 6.13, 1.35], [1.07, 4.37, 1.17], [1.1, 1.3, None],
    [1.218, 5.609, 1.348], [1.35, 3.06, 1.32], [1.4, 0.96, None],
    [1.95, 3.5, 1.54], [18.1, 21.6, 16.9], [6.95, 14.3, 6.58],
]
Silverbox_labels = ["out 1", "out 2", "out 3"]

CoupledElecDrives = [
    [0.062, 0.047], [0.085, 0.077], [0.087, 0.0696], [0.114, 0.138],
    [0.115, 0.074], [0.121, 0.097], [0.124, 0.194], [0.130, 0.185],
    [0.139, 0.111], [0.143, 0.100], [0.149, 0.120], [0.150, 0.092],
    [0.150, 0.167], [0.153, 0.132], [0.169, 0.117], [0.18, 0.159],
    [0.181, 0.191], [0.198, 0.158], [0.216, 0.110], [0.23, 0.137],
    [0.362, 0.370], [0.433, 0.179], [0.585, 0.962],
]
CED_labels = ["out 1", "out 2"]

ParallelWH = [
    [0.200, 0.397, 0.412, 0.469, 0.827], [1.75, 3.82, 5.61, 7.23, 10.5],
    [2.27, 4.02, 5.27, 9.91, 11.51], [15, 73.4, 164, 249, 328],
    [23.7, 21.6, 25.2, 40, 60.1], [25.5, 79.9, 132, 180, 224],
    [380, 396, 424, 455, 486], [4.32, 6.49, 9.49, 12.8, 16.9],
    [5.12, 7.29, 9.75, 12.6, 17.2], [8.48, 18.4, 27.5, 38, 51.1],
]
ParallelWH_labels = ["out 1", "out 2", "out 3", "out 4", "out 5"]

F16 = [
    [0.111, 0.264, 0.383, 0.0876, 0.191, 0.315], [0.119, 0.269, 0.389, 0.0998, 0.194, 0.339],
    [0.122, 0.268, 0.379, 0.0874, 0.187, 0.3], [0.262, 0.464, 0.603, 0.158, 0.279, 0.386],
    [0.285, 0.468, 0.595, 0.175, 0.288, 0.388], [0.289, 0.494, 0.634, 0.177, 0.297, 0.404],
    [0.289, 0.498, 0.643, 0.178, 0.298, 0.408], [0.315, 0.541, 0.699, 0.192, 0.335, 0.448],
    [0.422, 1.04, 1.38, 0.254, 0.66, 1.04],
]
F16_labels = ["out 1", "out 2", "out 3", "out 4", "out 5", "out 6"]

FineSteeringMirror = [
    [0.0866, 0.131, 0.169],
    [0.114, 0.216, 0.228],
]
FSM_labels = ["out 1", "out 2", "out 3"]

# ============================================================
# >>> FILL IN YOUR ASIA RESULTS HERE ONCE TRAINING IS DONE <<<
# Single-condition benchmarks: a single float.
# Multi-condition benchmarks: a list matching the number of conditions above.
# ============================================================
ASIA_RESULTS = {
    "Wiener-Hammerstein":      2.8150,   # single float
    "EMPS":                    4.1044,   # single float
    "Silverbox":               [0.91,3.78,0.83],   # e.g. [v1, v2, v3]
    "Coupled Electric Drives": [0.0956, 0.1003],   # e.g. [v1, v2]
    "Parallel WH":             [2.6890, 4.3642, 5.4269, 10.0683, 11.1642],   # e.g. [v1, v2, v3, v4, v5]
    "F16":                     [0.1107, 0.1935, 0.2930, 0.0628, 0.1183, 0.1809],   # e.g. [v1, v2, v3, v4, v5, v6]
    "Fine Steering Mirror":    [0.59,1.02,1.8],   # e.g. [v1, v2, v3]
}

BENCHMARKS = {
    "Wiener-Hammerstein":      dict(data=WH, labels=None),
    "EMPS":                    dict(data=EMPS, labels=None),
    "Silverbox":               dict(data=Silverbox, labels=Silverbox_labels),
    "Coupled Electric Drives": dict(data=CoupledElecDrives, labels=CED_labels),
    "Parallel WH":             dict(data=ParallelWH, labels=ParallelWH_labels),
    "F16":                     dict(data=F16, labels=F16_labels),
    # force_points=True: always plot dots, never a boxplot, regardless of BOX_MIN_N
    "Fine Steering Mirror":    dict(data=FineSteeringMirror, labels=FSM_labels, force_points=True),
}

# minimum number of non-missing literature values required to draw a boxplot;
# below this threshold, individual points are plotted instead
BOX_MIN_N = 4


def get_columns(data, n_cond):
    cols = [[] for _ in range(n_cond)]
    for row in data:
        for j in range(n_cond):
            v = row[j]
            if v is not None:
                cols[j].append(v)
    return [np.array(c, dtype=float) for c in cols]


def plot_panel(ax, name, spec, asia_val):
    data = spec["data"]
    labels = spec["labels"]
    force_points = spec.get("force_points", False)

    if labels is None:
        n_cond = 1
        cols = [np.array(data, dtype=float)]
        xlabels = [""]
    else:
        n_cond = len(labels)
        cols = get_columns(data, n_cond)
        xlabels = labels

    for i, vals in enumerate(cols):
        n = len(vals)
        use_box = (n >= BOX_MIN_N) and not force_points
        if use_box:
            bp = ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                             showfliers=False, medianprops=dict(color='red', linewidth=1.4))
            for patch in bp['boxes']:
                patch.set_facecolor('#cfe0f3')
                patch.set_alpha(0.85)
        jitter = np.random.normal(0, 0.05, size=n)
        ax.scatter(np.full(n, i) + jitter, vals, color='gray', s=16, zorder=3, alpha=0.75)

    if isinstance(asia_val, str) and asia_val.upper() == "TBD":
        ax.text(0.5, 0.5, "ASIA: TBD", transform=ax.transAxes, ha='center', va='center',
                 color='darkorange', fontweight='bold',
                bbox=dict(boxstyle="round", facecolor='white', edgecolor='darkorange'))
    else:
        asia_arr = [asia_val] if n_cond == 1 else asia_val
        ax.scatter(range(n_cond), asia_arr, color='crimson', s=85, zorder=5,
                   edgecolor='black', linewidth=0.6)

    ax.set_xticks(range(n_cond))
    ax.set_xticklabels(xlabels,  rotation=15 if n_cond > 1 else 0)
    ax.set_yscale('log')
    ax.set_title(name)
    ax.grid(axis='y', which='both', alpha=0.25)


def main():
    n_benchmarks = len(BENCHMARKS)
    ncols = 4
    nrows = int(np.ceil((n_benchmarks + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows))
    axes = axes.flatten()

    for ax, (name, spec) in zip(axes, BENCHMARKS.items()):
        plot_panel(ax, name, spec, ASIA_RESULTS[name])

    for ax in axes[n_benchmarks:]:
        ax.axis('off')

    legend_ax = axes[n_benchmarks]
    legend_ax.axis('off')
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Literature value'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', markeredgecolor='black', markersize=10, label='ASIA (ours)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#cfe0f3', alpha=0.85, label=f'Boxplot (n\u2265{BOX_MIN_N})'),
    ]
    legend_ax.legend(handles=handles, loc='center',  frameon=False)

    fig.supylabel("Test RMSE (log scale)")
    plt.suptitle("Test RMSE comparison against literature, per benchmark", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('output/benchmark_comparison_all.pdf', dpi=200)
    print("Saved to output/benchmark_comparison_all.pdf")


if __name__ == "__main__":
    main()