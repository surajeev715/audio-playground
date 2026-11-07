"""
Plots the recordings from the Audio Playground.

Reads every CSV in the recordings folder. Each file counts as one measurement,
named after its file name (e.g. phone-chrome-quiet.csv -> "phone-chrome-quiet").
The background level is worked out per file from its "silence" segments, or from
the quietest 10 % of the file if it has none. Then it draws:

  1. time_series.png  One row per file, one column per label.
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
    frames = []
    for f in files:
        frame = pd.read_csv(f)
        # Each CSV file is one measurement: its file name (without .csv) is used as the
        # row name in the plots, instead of the device name typed into the page.
        frame["device"] = f.stem
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} file(s), {len(data)} rows: {', '.join(f.stem for f in files)}")
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
        figsize=(4 * len(labels) + 1.5, 2.6 * len(devices) + 1),
        squeeze=False, sharex=True, sharey=True
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
                ax.set_title(label, fontsize=13, fontweight="bold")
            if c == 0:
                # Device name as a row header, left of the plots
                ax.set_ylabel(device, fontsize=12, fontweight="bold", rotation=0,
                              ha="right", va="center", labelpad=12)

    # Axis descriptions once for the whole figure instead of in every row
    fig.supxlabel("time in segment (s)")
    fig.supylabel("RMS as a multiple of the background (log scale)", x=0.005)
    fig.suptitle("Each recording relative to its device's background (dashed line = background)", fontsize=13)
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

    # Two panels on top of each other, so long device names have the full width
    width = max(8, 1.4 * len(devices) + 3)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(width, 8), sharex=True)
    for ax, column, title, ylabel in [
        (top, "max_rms", "Loudest moment per recording, absolute RMS", "RMS"),
        (bottom, "max_ratio", "Loudest moment per recording, relative to the device's background", "RMS / background"),
    ]:
        for i, label in enumerate(labels):
            rows = per_segment[per_segment["label"] == label]
            offset = (i - (len(labels) - 1) / 2) * 0.15
            xs = [devices.index(d) + offset for d in rows["device"]]
            ax.scatter(xs, rows[column], label=label, color=LABEL_COLORS.get(label, "#555555"), s=40, alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", axis="y", alpha=0.25)
        # Light separators between devices
        for x in range(1, len(devices)):
            ax.axvline(x - 0.5, color="#dddddd", linewidth=0.8)

    bottom.set_xticks(range(len(devices)))
    bottom.set_xticklabels(devices, rotation=30, ha="right")
    top.legend(loc="upper left", bbox_to_anchor=(1.01, 1), title="label")

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