"""
Finds the sound events in the recordings and tests the clap rule on them.

Uses the same CSVs and the same background/ratio calculation as plot.py.

Part 1, event features
    Every sound that rises above EVENT_GATE is one event. For each event it measures:
      peak_ratio     loudest moment, as a multiple of the background
      onset_jump     how much the ratio rose in one step at the start
                     (ratio of the first loud block / the block before it)
      half_peak_ms   how long it stays above half of its own peak
    Saves events.csv and events.png (onset jump vs. time above half peak, per label).

Part 2, clap rule simulation
    Runs the same three states as ClapSensor (Idle -> Measuring -> Echo) over every
    recording and counts the detected claps per file and label.
    Clap segments should show your number of claps, all other labels 0.

Usage:
    python events.py                  (reads ../recordings)
    python events.py path/to/folder

Change the values in the CLAP RULE section and run it again to try other thresholds.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot import LABEL_COLORS, add_ratios, load_recordings, ordered_labels

# ---------- Event features ----------
EVENT_GATE = 8.0           # a block above this ratio starts an event
EVENT_MAX_SECONDS = 3.0    # stop following an event after this long

# ---------- CLAP RULE (same names as in ClapSensor, change and re-run) ----------
GATE = 15.0                          # ratio needed to start measuring
MIN_ONSET_JUMP = 5.0                 # ratio / previous ratio needed at the start
CLAP_MAX_DURATION_IN_SECONDS = 0.15  # max time until below half its peak
ECHO_END_RATIO = 3.0                 # echo ends when the ratio falls below this ...
CLAP_ECHO_MAX_DURATION_IN_SECONDS = 0.6  # ... or after this long at the latest

# Unity reads once per frame, not every 10 ms. STEP = 1 uses every 10 ms block
# (like the gap-free worklet), STEP = 2 or 3 roughly simulates 50 or 33 FPS.
STEP = 1


# ---------- Part 1: event features ----------
def find_events(segment: pd.DataFrame) -> list:
    """Walks through one recorded segment and returns one dict per sound event."""
    times = segment["time_ms"].to_numpy()
    ratios = segment["rms_ratio"].to_numpy()
    block_ms = times[1] - times[0] if len(times) > 1 else 10.0
    max_blocks = int(EVENT_MAX_SECONDS * 1000 / block_ms)

    events = []
    i = 1
    while i < len(ratios):
        if ratios[i] >= EVENT_GATE and ratios[i - 1] < EVENT_GATE:
            start = i
            onset_jump = ratios[i] / max(ratios[i - 1], 1e-6)

            # Follow the event while it stays above the gate
            end = start
            while end + 1 < len(ratios) and ratios[end + 1] >= EVENT_GATE and end - start < max_blocks:
                end += 1

            window = ratios[start:end + 1]
            peak = window.max()
            half_peak_blocks = int((window >= peak / 2).sum())

            events.append({
                "start_s": times[start] / 1000,
                "peak_ratio": peak,
                "onset_jump": onset_jump,
                "half_peak_ms": half_peak_blocks * block_ms,
                "above_gate_ms": (end - start + 1) * block_ms,
            })
            i = end + 1
        else:
            i += 1
    return events


def collect_events(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (device, label, segment_id), segment in data.groupby(["device", "label", "segment"]):
        segment = segment.sort_values("time_ms")
        for event in find_events(segment):
            rows.append({"file": device, "label": label, "segment": segment_id, **event})
    return pd.DataFrame(rows)


def plot_events(events: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for label in ordered_labels(events):
        rows = events[events["label"] == label]
        ax.scatter(rows["half_peak_ms"], rows["onset_jump"], label=f"{label} ({len(rows)})",
                   color=LABEL_COLORS.get(label, "#555555"), s=40, alpha=0.8)

    # The current clap rule as a box: short and sharp = top left
    ax.axhline(MIN_ONSET_JUMP, color="#e0703a", linestyle="--", linewidth=1)
    ax.axvline(CLAP_MAX_DURATION_IN_SECONDS * 1000, color="#e0703a", linestyle="--", linewidth=1)
    ax.text(CLAP_MAX_DURATION_IN_SECONDS * 1000, MIN_ONSET_JUMP,
            "  clap rule: left of and above the dashed lines", color="#e0703a", fontsize=9, va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time above half of its own peak (ms)")
    ax.set_ylabel("onset jump (ratio / previous ratio)")
    ax.set_title("Every sound event in the recordings")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="label (events)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def print_feature_summary(events: pd.DataFrame) -> None:
    print("\nEvent features per label (min / median / max):")
    for label in ordered_labels(events):
        rows = events[events["label"] == label]
        print(f"\n  {label}: {len(rows)} events")
        for column in ["peak_ratio", "onset_jump", "half_peak_ms"]:
            values = rows[column]
            print(f"    {column:<13} {values.min():>9.1f} / {values.median():>9.1f} / {values.max():>9.1f}")


# ---------- Part 2: clap rule simulation ----------
def simulate_clap_rule(ratios, times_s) -> list:
    """Same three states as ClapSensor. Returns the times (s) of detected claps."""
    state = "idle"
    previous_ratio = 1.0
    peak_ratio = 0.0
    start_time = 0.0
    claps = []

    for ratio, now in zip(ratios, times_s):
        if state == "idle":
            if ratio >= GATE and ratio / max(previous_ratio, 1e-4) >= MIN_ONSET_JUMP:
                state = "measuring"
                peak_ratio = ratio
                start_time = now

        elif state == "measuring":
            peak_ratio = max(peak_ratio, ratio)
            duration = now - start_time
            if ratio < peak_ratio / 2:
                if duration <= CLAP_MAX_DURATION_IN_SECONDS:
                    claps.append(start_time)
                    state = "echo"
                    start_time = now
                else:
                    state = "idle"          # too long: blow or talk
            elif duration > CLAP_MAX_DURATION_IN_SECONDS * 3:
                state = "idle"              # stays loud: give up

        elif state == "echo":
            faded = ratio < ECHO_END_RATIO
            timeout = now - start_time >= CLAP_ECHO_MAX_DURATION_IN_SECONDS
            if faded or timeout:
                state = "idle"

        previous_ratio = ratio
    return claps


def run_simulation(data: pd.DataFrame) -> None:
    results = []
    for (device, label, segment_id), segment in data.groupby(["device", "label", "segment"]):
        segment = segment.sort_values("time_ms").iloc[::STEP]
        claps = simulate_clap_rule(segment["rms_ratio"].to_numpy(), (segment["time_ms"] / 1000).to_numpy())
        results.append({"file": device, "label": label, "claps": len(claps)})

    table = pd.DataFrame(results).pivot_table(index="file", columns="label", values="claps", aggfunc="sum")
    labels = [l for l in ordered_labels(data) if l in table.columns]
    print(f"\nClap rule: GATE={GATE}, MIN_ONSET_JUMP={MIN_ONSET_JUMP}, "
          f"CLAP_MAX_DURATION={CLAP_MAX_DURATION_IN_SECONDS}s, STEP={STEP}")
    print("Detected claps per recording (clap column = your claps, all others should be 0):")
    print(table[labels].fillna(0).astype(int).to_string())


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else script_dir.parent / "recordings"
    out_dir = script_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    data = add_ratios(load_recordings(folder))

    events = collect_events(data)
    if events.empty:
        print("\nNo events found above EVENT_GATE.")
    else:
        events.to_csv(out_dir / "events.csv", index=False)
        print(f"\nSaved {out_dir / 'events.csv'} ({len(events)} events)")
        plot_events(events, out_dir / "events.png")
        print_feature_summary(events)

    run_simulation(data)


if __name__ == "__main__":
    main()