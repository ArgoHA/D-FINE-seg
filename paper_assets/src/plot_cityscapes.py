"""
Cityscapes headline figure: accuracy vs latency, bubble size = params.
One panel per task (detection / instance seg / semantic seg). D-FINE-seg lands
top-left (more accurate, faster) and smallest across all three.
Data mirrors the Cityscapes tables in README.md.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 1.2

# framework -> color
C = {"D-FINE-seg": "#e85d00", "YOLO26": "#0c27c4", "RF-DETR": "#6412a3"}

# (framework, label, params M, latency e2e ms, metric)
PANELS = [
    (
        "Detection",
        "F1",
        [
            ("D-FINE-seg", "S", 10.29, 2.0, 0.703),
            ("YOLO26", "M", 21.79, 3.03, 0.691),
            ("RF-DETR", "M", 33.39, 10.2, 0.673),
        ],
    ),
    (
        "Instance segmentation",
        "F1",
        [
            ("D-FINE-seg", "S", 11.87, 4.1, 0.661),
            ("YOLO26", "M", 26.98, 5.24, 0.599),
            ("RF-DETR", "M", 35.4, 16.33, 0.62),
        ],
    ),
    (
        "Semantic segmentation",
        "mIoU",
        [
            ("D-FINE-seg", "S", 8.02, 1.79, 0.728),
            ("D-FINE-seg", "M", 16.0, 2.24, 0.753),
            ("YOLO26", "M", 14.32, 3.08, 0.733),
            ("YOLO26", "L", 17.87, 3.56, 0.739),
        ],
    ),
]


def params_to_size(p):
    # bubble area scaled to params (M); keep dots readable
    return 90 + p * 55


fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), dpi=150)

for ax, (task, metric, rows) in zip(axes, PANELS):
    xs = [r[3] for r in rows]
    ys = [r[4] for r in rows]

    for fw, size, params, lat, val in rows:
        ax.scatter(
            lat,
            val,
            s=params_to_size(params),
            color=C[fw],
            alpha=0.9,
            edgecolors="white",
            linewidths=1.6,
            zorder=5 if fw == "D-FINE-seg" else 4,
        )
        ax.annotate(
            f"{size}\n{params:.0f}M",
            (lat, val),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            zorder=6,
        )

    x_pad = (max(xs) - min(xs)) * 0.16 + 0.3
    y_pad = (max(ys) - min(ys)) * 0.28 + 0.004
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    ax.set_title(f"{task}", fontsize=14, fontweight="medium", pad=10)
    ax.set_xlabel("Latency e2e (ms)  <-- faster", fontsize=12, fontweight="medium")
    ax.set_ylabel(f"{metric}  --> better", fontsize=12, fontweight="medium")
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

# shared framework legend
handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markersize=11,
        markerfacecolor=c,
        markeredgecolor="white",
        label=fw,
    )
    for fw, c in C.items()
]
fig.legend(
    handles=handles,
    loc="upper center",
    ncol=3,
    fontsize=12,
    framealpha=0.95,
    edgecolor="gray",
    fancybox=True,
    bbox_to_anchor=(0.5, 0.055),
)
fig.suptitle(
    "Cityscapes val - accuracy vs latency  (TensorRT FP16, RTX 5070 Ti; bubble = params)",
    fontsize=15,
    y=0.99,
)

plt.tight_layout(rect=[0, 0.06, 1, 0.96])
out = "assets/cityscapes_benchmark.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
print(f"Saved: {out}")
