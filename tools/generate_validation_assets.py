from pathlib import Path

import csv

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "sample_data"


TIME_POINTS = [0, 6, 12, 18, 24, 30]
SURVIVAL_A = [1.00, 0.95, 0.89, 0.82, 0.76, 0.70]
SURVIVAL_B = [1.00, 0.88, 0.76, 0.63, 0.50, 0.38]
RISK_A = [120, 114, 104, 92, 78, 65]
RISK_B = [120, 106, 91, 73, 56, 41]


def write_truth_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_months",
                "survival_treatment_a",
                "survival_treatment_b",
                "risk_treatment_a",
                "risk_treatment_b",
            ]
        )
        for row in zip(TIME_POINTS, SURVIVAL_A, SURVIVAL_B, RISK_A, RISK_B):
            writer.writerow(row)


def build_figure():
    fig = plt.figure(figsize=(10.8, 7.4), facecolor="white")
    grid = GridSpec(7, 1, height_ratios=[5.2, 0.15, 0.7, 0.7, 0.15, 0.4, 0.2], hspace=0.0)

    ax = fig.add_subplot(grid[0])
    ax.step(TIME_POINTS, SURVIVAL_A, where="post", linewidth=2.6, color="#2b6cb0", label="Treatment A")
    ax.step(TIME_POINTS, SURVIVAL_B, where="post", linewidth=2.6, color="#c53030", label="Treatment B")
    ax.scatter([9, 21], [0.92, 0.79], color="#2b6cb0", marker="+", s=90, linewidths=1.6)
    ax.scatter([15, 27], [0.69, 0.45], color="#c53030", marker="+", s=90, linewidths=1.6)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(TIME_POINTS)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel("Time (months)", fontsize=11)
    ax.set_ylabel("Survival probability", fontsize=11)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(
        "Synthetic Kaplan-Meier Example: Treatment A vs Treatment B\n"
        "Expected direction: A performs better than B",
        fontsize=13,
        pad=12,
    )

    table_title_ax = fig.add_subplot(grid[2])
    table_title_ax.axis("off")
    table_title_ax.text(0.0, 0.6, "Number at risk", fontsize=11, weight="bold")

    row_a_ax = fig.add_subplot(grid[3])
    row_b_ax = fig.add_subplot(grid[5])
    for axis, label, values, color in [
        (row_a_ax, "Treatment A", RISK_A, "#2b6cb0"),
        (row_b_ax, "Treatment B", RISK_B, "#c53030"),
    ]:
        axis.set_xlim(-8, 30)
        axis.set_ylim(0, 1)
        axis.axis("off")
        axis.text(-7.4, 0.48, label, fontsize=10, color=color, ha="left", va="center")
        for time_value, risk_value in zip(TIME_POINTS, values):
            axis.text(time_value, 0.48, str(risk_value), fontsize=10, ha="center", va="center")

    footer_ax = fig.add_subplot(grid[6])
    footer_ax.axis("off")
    footer_ax.text(
        0.0,
        0.25,
        "Goal for smoke test: detect 2 curves, parse risk table, reconstruct IPD, report HR > 1 for B vs A direction in app.",
        fontsize=9,
        color="#444444",
    )
    return fig


def main():
    OUT_DIR.mkdir(exist_ok=True)
    png_path = OUT_DIR / "km_minimal_sample.png"
    pdf_path = OUT_DIR / "km_minimal_sample.pdf"
    truth_path = OUT_DIR / "km_minimal_truth.csv"

    figure = build_figure()
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    write_truth_csv(truth_path)
    print("Wrote:")
    print(png_path)
    print(pdf_path)
    print(truth_path)


if __name__ == "__main__":
    main()
