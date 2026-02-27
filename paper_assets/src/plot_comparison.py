"""
Model Comparison Plot: F1-score vs Latency
Reads data from a CSV file
"""

import matplotlib.pyplot as plt
import pandas as pd

# Configure plot style
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 1.2


def plot_comparison(csv_path: str, output_name: str = "model_comparison", task: str = "segment"):
    """
    Plot F1-score vs Latency from a CSV file.

    Expected CSV columns:
    - 'latency' or 'Latency': latency values in ms
    - 'f1' or 'F1-score' or 'f1_score' or 'accuracy': metric values
    - 'model' or 'variant' (optional): labels for each point
    """

    # Read CSV
    df = pd.read_csv(csv_path)

    # Normalize column names (case-insensitive)
    df.columns = df.columns.str.lower().str.strip()

    # Find latency column
    latency_col = None
    for col in ["latency", "latency_ms", "time", "inference_time"]:
        if col in df.columns:
            latency_col = col
            break
    if latency_col is None:
        raise ValueError(f"Could not find latency column. Available: {list(df.columns)}")

    # Find F1/accuracy column
    f1_col = None
    for col in ["f1-score", "f1_score", "f1", "accuracy", "acc", "map", "ap"]:
        if col in df.columns:
            f1_col = col
            break
    if f1_col is None:
        raise ValueError(f"Could not find F1-score/accuracy column. Available: {list(df.columns)}")

    # Find model column
    model_col = None
    for col in ["model", "variant", "name", "label"]:
        if col in df.columns:
            model_col = col
            break

    if model_col is None:
        raise ValueError(f"Could not find model column. Available: {list(df.columns)}")

    # Extract base model name and variant from "Model variant" format
    # e.g., "D-FINE-seg n" -> base="D-FINE-seg", variant="n"
    def parse_model_name(name):
        parts = str(name).rsplit(" ", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return name, ""

    df["base_model"] = df[model_col].apply(lambda x: parse_model_name(x)[0])
    df["variant"] = df[model_col].apply(lambda x: parse_model_name(x)[1])
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    # Colors and markers for different models
    colors = ["#ff6600", "#0b23a9"]
    markers = ["o", "s", "D", "^", "v"]

    # Get unique base models
    base_models = df["base_model"].unique()

    for idx, base_model in enumerate(base_models):
        model_df = df[df["base_model"] == base_model].copy()
        if base_model == "D-FINE-seg" and task == "detect":
            base_model = "D-FINE"

        # Sort by latency
        model_df = model_df.sort_values(by=latency_col)

        latency = model_df[latency_col].values
        f1 = model_df[f1_col].values
        variants = model_df["variant"].values

        print(f1)

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        # zorder: first model (D-FINE, orange) on top
        z = 20 - idx * 5

        # Plot line connecting points
        ax.plot(
            latency,
            f1,
            color=color,
            linewidth=2.5,
            linestyle="-",
            zorder=z,
            label=base_model,
        )

        # Plot markers
        ax.scatter(
            latency,
            f1,
            color=color,
            s=100,
            marker=marker,
            edgecolors="white",
            linewidths=1.5,
            zorder=z + 1,
        )

        # Add variant labels
        for lat, f1_val, var in zip(latency, f1, variants):
            ax.annotate(
                str(var),
                (lat, f1_val),
                xytext=(lat - 0.03, f1_val + 0.007),
                fontsize=12,
                fontweight="medium",
                color=color,
                zorder=z + 2,
            )

    # Get global min/max for axis limits
    latency_all = df[latency_col].values
    f1_all = df[f1_col].values

    # Format task name for display
    task_display = {"segment": "Segmentation", "detect": "Detection"}.get(
        task.lower(), task.capitalize()
    )

    # Labels and title
    ax.set_xlabel("Latency (ms)", fontsize=13, fontweight="medium")
    ax.set_ylabel("F1-score", fontsize=13, fontweight="medium")
    ax.set_title(f"F1-score vs Latency, {task_display} on VisDrone dataset", fontsize=15, pad=15)

    # Grid
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc="lower right", fontsize=12, framealpha=0.95, edgecolor="gray", fancybox=True)

    # Auto axis limits with padding
    x_pad = (latency_all.max() - latency_all.min()) * 0.1 or 1
    y_pad = (f1_all.max() - f1_all.min()) * 0.1 or 0.01
    ax.set_xlim(latency_all.min() - x_pad, latency_all.max() + x_pad)
    ax.set_ylim(f1_all.min() - y_pad, f1_all.max() + y_pad)

    # Ticks
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()

    # Save
    plt.savefig(
        f"{output_name}.png", dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none"
    )

    print(f"✓ Saved: {output_name}.png")


if __name__ == "__main__":
    for task in ["detect"]:
        csv_path = "visdrone.csv"
        plot_comparison(csv_path, f"comparison_{task}", task)
