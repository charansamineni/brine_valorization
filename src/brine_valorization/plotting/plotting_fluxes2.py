import re
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# ============================================================
# H5 files
# ============================================================

h5_file_70gL = "voltageub1200_recovery_sweep_70gL_08-06_0930.h5"
h5_file_150gL = "voltageub1200_recovery_sweep_150gL_08-06_0930.h5"

# (feed label in g/L, h5 file path)
CASES = [
    (70, h5_file_70gL),
    (150, h5_file_150gL),
]

# How many NaCl Recovery curves to draw per plot. choose_recoveries() below
# automatically restricts itself to whichever recoveries actually have real
# (non-NaN) data in a given file, so the 70 g/L case will span its full
# 0.5-0.85 solved range and the 150 g/L case will automatically stop wherever
# it actually stops converging (around ~0.74) -- no per-case range needs to
# be hardcoded here.
N_RECOVERY_CURVES = 5

STREAMS = ["Diluate", "Acidate", "Basate"]

# Component groupings for separate figures: H2O alone, Na+/Cl- together,
# H+/OH- together.
COMPONENT_GROUPS = {
    "H2O": ["H2O"],
    "NaCl": ["Na_+", "Cl_-"],
    "Water Ions": ["H_+", "OH_-"],
}

# Linestyle per component -- used to distinguish the two components within
# a shared graph (color is reserved for NaCl Recovery).
LINESTYLES = {
    "H2O": "-",
    "Na_+": "-",
    "Cl_-": "--",
    "H_+": "-",
    "OH_-": "--",
}

# cividis is a perceptually-uniform, colorblind-safe colormap.
CMAP = plt.cm.cividis


# ============================================================
# Load ALL mass transfer datasets from an h5 results file
# ============================================================

# Matches dataset paths like:
#   outputs/fs.bpmed.bpmed[0].diluate.mass_transfer_term[0.0,0.2,Liq,Na_+]/value
MASS_TRANSFER_PATTERN = re.compile(
    r"outputs/fs\.bpmed\.bpmed\[0\]\."
    r"(diluate|acidate|basate)"
    r"\.mass_transfer_term"
    r"\[0\.0,([0-9.]+),Liq,([^\]]+)\]/value"
)


def load_mass_transfer_data(h5_file):
    """Parse every diluate/acidate/basate mass_transfer_term dataset out of
    an h5 sweep-results file into one long-format DataFrame, with channel
    position already converted from relative (0-1) to physical (m) using
    each row's own Cell Length -- since cell length can differ between
    NaCl Recovery points.
    """
    rows = []

    with h5py.File(h5_file, "r") as f:
        recoveries = f["sweep_params/NaCl Recovery/value"][:]
        cell_lengths = f["outputs/fs.bpmed.bpmed[0].cell_length/value"][:]

        def visitor(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return

            match = MASS_TRANSFER_PATTERN.fullmatch(name)
            if match is None:
                return

            stream = match.group(1).capitalize()
            relative_position = float(match.group(2))
            component = match.group(3)

            values = obj[:]

            for i, value in enumerate(values):
                rows.append({
                    "NaCl Recovery": recoveries[i],
                    "Stream": stream,
                    "Component": component,
                    "Relative Position": relative_position,
                    "Cell Length": cell_lengths[i],
                    "Channel Position": relative_position * cell_lengths[i],
                    "Mass Transfer": value,
                })

        f.visititems(visitor)

    return pd.DataFrame(rows)


# ============================================================
# Choose recovery values to plot
# ============================================================

def choose_recoveries(df, n=N_RECOVERY_CURVES):
    """
    Choose n recovery values spanning the successfully-solved range.

    A recovery value is only considered valid if EVERY row for it (across
    every stream, component, and channel position) is non-NaN. This means
    the selection automatically adapts to each feed case's actual feasible
    range -- e.g. a 150 g/L file that stops converging around 0.74 will
    only ever offer recoveries up to that point, with no hardcoded cutoff
    needed per case. Called once per feed case (on the full DataFrame, not
    filtered to a single stream) so that every stream/component-group plot
    for that feed uses the exact same 5 recovery values and colors.
    """
    valid = df.groupby("NaCl Recovery")["Mass Transfer"].apply(
        lambda x: x.notna().all()
    )
    recoveries = np.sort(valid[valid].index.values)

    if len(recoveries) == 0:
        return []

    desired = np.linspace(recoveries.min(), recoveries.max(), n)

    selected = []
    for r in desired:
        nearest = recoveries[np.argmin(np.abs(recoveries - r))]
        if nearest not in selected:
            selected.append(nearest)

    return selected


# ============================================================
# Plotting
# ============================================================

def plot_mass_transfer(df, stream, group_name, components, recoveries, feed):
    """Plot mass_transfer_term vs. channel position for one stream and one
    component group (H2O alone, Na+/Cl- together, or H+/OH- together).

    One line per NaCl Recovery value (color, colorblind-safe continuous
    scale); one linestyle per component when more than one component is
    plotted together.
    """
    df_stream = df[df["Stream"] == stream]

    if len(recoveries) == 0:
        print(f"No converged data for {stream} / {group_name} ({feed} g/L) -- skipping.")
        return None

    norm = mcolors.Normalize(vmin=min(recoveries), vmax=max(recoveries))

    fig, ax = plt.subplots(figsize=(9, 6))

    any_data_plotted = False

    for recovery in recoveries:
        df_rec = df_stream[np.isclose(df_stream["NaCl Recovery"], recovery)]
        color = CMAP(norm(recovery))

        for comp in components:
            df_comp = (
                df_rec[df_rec["Component"] == comp]
                .sort_values("Channel Position")
            )
            if df_comp.empty:
                continue

            ax.plot(
                df_comp["Channel Position"],
                df_comp["Mass Transfer"],
                color=color,
                linestyle=LINESTYLES.get(comp, "-"),
                linewidth=2,
                alpha=0.9,
            )
            any_data_plotted = True

    if not any_data_plotted:
        plt.close(fig)
        print(f"No data found for {stream} / {group_name} ({feed} g/L) -- skipping.")
        return None

    ax.axhline(y=0, color="grey", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Channel Position (m)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Mass Transfer Term", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(
        f"{stream} Channel \u2014 {group_name} Mass Transfer, {feed} g/L Feed",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.6)

    # Colorbar for NaCl Recovery (color-only encoding -> works for any CVD type)
    sm = cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("NaCl Recovery (fraction)", rotation=270, labelpad=15)

    # Linestyle legend -- only needed when two components share a plot
    if len(components) > 1:
        legend_handles = [
            plt.Line2D(
                [0], [0],
                color="black",
                linestyle=LINESTYLES.get(c, "-"),
                linewidth=2,
                label=c,
            )
            for c in components
        ]
        ax.legend(
            handles=legend_handles,
            loc="best",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()
    return fig


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

    for feed, h5_file in CASES:
        print(f"\n=== Loading {feed} g/L case from {h5_file} ===")
        df = load_mass_transfer_data(h5_file)

        if df.empty:
            print(
                f"No mass_transfer_term datasets found in {h5_file} "
                f"-- check the file path or the naming pattern."
            )
            continue

        recoveries = choose_recoveries(df, n=N_RECOVERY_CURVES)
        print(f"Selected {len(recoveries)} recovery values for {feed} g/L: "
              f"{[round(r, 4) for r in recoveries]}")

        for stream in STREAMS:
            for group_name, components in COMPONENT_GROUPS.items():
                fig = plot_mass_transfer(df, stream, group_name, components, recoveries, feed)
                if fig is None:
                    continue

                # safe_group = group_name.replace(" ", "")
                # fname = f"MassTransfer_{stream}_{safe_group}_{feed}gL_{timestamp}.png"
                # fig.savefig(fname, dpi=300, bbox_inches="tight")
                # plt.close(fig)
                # print(f"Saved {fname}")

    print("\nDone.")