"""
Plots the recordings from the Audio Playground.

Reads every CSV in the recordings folder, works out each device's background
level from its "silence" segments, and draws:

  1. time_series.png  One row per device, one column per label.
                      Every segment as RMS ratio over time (log scale).
  2. summary.png      Per segment: the loudest moment, once as absolute RMS
                      and once as ratio to the device's background.
                      If the ratio idea works, the right side lines up across
                      devices even when the left side does not.

Usage:
    python plot.py                  (reads ../recordings)
    python plot.py path/to/folder   (reads another folder)

Needs: pip install pandas matplotlib
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LABEL_ORDER = ["silence", "clap", "blow", "talk"]
LABEL_COLORS = {"silence": "#8a8a8a", "clap": "#e0703a", "blow": "#2f6fde", "talk": "#2e8b57"}


def load_recordings(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        sys.exit(f"No CSV files found in {folder}")
    frames = [pd.read_csv(f) for f in files]
    data = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} file(s), {len(data)} rows, devices: {', '.join(data['device'].unique())}")
    return data


def add_ratios(data: pd.DataFrame) -> pd.DataFrame:
    """Background = median RMS of the device's silence segments.
    Without silence recordings, the quietest 10 % of all its values are used instead."""
    data = data.copy()
    for device, rows in data.groupby("device"):
        silence = rows[rows["label"] == "silence"]["rms"]
        if len(silence) > 0:
            background = silence.median()
        else:
            background = rows["rms"].quantile(0.10)
            print(f"  {device}: no silence segment, using the quietest 10 % as background")
        background = max(background, 1e-6)
        data.loc[rows.index, "background"] = background
        print(f"  {device}: background RMS = {background:.6f}")

    data["rms_ratio"] = data["rms"] / data["background"]
    data["peak_ratio"] = data["peak"] / data["background"]
    return data


def ordered_labels(data: pd.DataFrame) -> list:
    present = list(data["label"].unique())
    return [l for l in LABEL_ORDER if l in present] + [l for l in present if l not in LABEL_ORDER]


def plot_time_series(data: pd.DataFrame, out: Path) -> None:
    devices = sorted(data["device"].unique())
    labels = ordered_labels(data)

    fig, axes = plt.subplots(
        len(devices), len(labels),
        figsize=(4 * len(labels), 2.8 * len(devices)),
        squeeze=False, sharey=True
    )

    for r, device in enumerate(devices):
        for c, label in enumerate(labels):
            ax = axes[r][c]
            rows = data[(data["device"] == device) & (data["label"] == label)]
            for _, segment in rows.groupby("segment"):
                ax.plot(segment["time_ms"] / 1000, segment["rms_ratio"],
                        color=LABEL_COLORS.get(label, "#555555"), linewidth=0.9, alpha=0.8)
            ax.axhline(1, color="#999999", linestyle="--", linewidth=0.8)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            if r == 0:
                ax.set_title(label)
            if c == 0:
                ax.set_ylabel(f"{device}\nRMS / background")
            if r == len(devices) - 1:
                ax.set_xlabel("time in segment (s)")

    fig.suptitle("RMS as a multiple of each device's background (dashed line = background)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_summary(data: pd.DataFrame, out: Path) -> None:
    per_segment = (
        data.groupby(["device", "label", "segment"])
        .agg(max_rms=("rms", "max"), max_ratio=("rms_ratio", "max"))
        .reset_index()
    )
    devices = sorted(per_segment["device"].unique())
    labels = ordered_labels(per_segment)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, column, title in [
        (left, "max_rms", "Loudest moment, absolute RMS"),
        (right, "max_ratio", "Loudest moment, RMS / background"),
    ]:
        for i, label in enumerate(labels):
            rows = per_segment[per_segment["label"] == label]
            offset = (i - (len(labels) - 1) / 2) * 0.15
            xs = [devices.index(d) + offset for d in rows["device"]]
            ax.scatter(xs, rows[column], label=label, color=LABEL_COLORS.get(label, "#555555"), s=36, alpha=0.85)
        ax.set_xticks(range(len(devices)))
        ax.set_xticklabels(devices)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", axis="y", alpha=0.25)
    right.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    print("\nMedian of the loudest ratio per segment:")
    table = per_segment.pivot_table(index="device", columns="label", values="max_ratio", aggfunc="median")
    print(table[[l for l in labels if l in table.columns]].round(1).to_string())


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else script_dir.parent / "recordings"
    out_dir = script_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    data = add_ratios(load_recordings(folder))
    plot_time_series(data, out_dir / "time_series.png")
    plot_summary(data, out_dir / "summary.png")


if __name__ == "__main__":
    main()
