import pandas as pd
import h5py
import matplotlib.pyplot as plt
from datetime import datetime

from sympy import product


#how LCOW is calculated
# fs.costing.total_capital_cost*fs.costing.capital_recovery_factor 
# + fs.costing.total_operating_cost) / (31557600.000000004*(s/a)*fs.total_product_water*fs.costing.utilization_factor)


h5_file_70gL = "voltageub1200_recovery_sweep_70gL_08-06_0930.h5"
h5_file_150gL = "voltageub1200_recovery_sweep_150gL_08-06_0930.h5"

def load_h5_results(h5_file):

    with h5py.File(h5_file, "r") as f:

        df = pd.DataFrame({
            "NaCl Recovery":
                f["sweep_params/NaCl Recovery/value"][:],

            "Capital Recovery Factor":
                f["outputs/fs.costing.capital_recovery_factor/value"][:],

            "Total Capital Cost":
                f["outputs/fs.costing.total_capital_cost/value"][:],

            "Total Operating Cost":
                f["outputs/fs.costing.total_operating_cost/value"][:],

            "Utilization Factor":
                f["outputs/fs.costing.utilization_factor/value"][:],

            "Flow Mass Product[bpmed_HCl]":
                f["outputs/fs.bpmed.flow_mass_product[bpmed,HCl]/value"][:],

            "Flow Mass Product[bpmed_NaOH]":
                f["outputs/fs.bpmed.flow_mass_product[bpmed,NaOH]/value"][:],
            "Voltage Applied":
                f["outputs/fs.bpmed.bpmed[0].voltage_applied[0.0]/value"] [:],
            "Cell Length":
                f["outputs/fs.bpmed.bpmed[0].cell_length/value"] [:],
            "Cell Width":
                f["outputs/fs.bpmed.bpmed[0].cell_width/value"] [:],
            "Cell Triplet Number":
                f["outputs/fs.bpmed.bpmed[0].cell_triplet_num/value"] [:],
            
        })

    return df
 
 
def add_normalized_costs(df, product="HCl"):
    df = df.copy()
 
    crf = df["Capital Recovery Factor"]
    total_capex = df["Total Capital Cost"]      # $ (at fs level)
    total_opex = df["Total Operating Cost"]      # $/yr, total fixed opex + total variable opex (at fs level)
 #   total_prod_water = df["Total Product Water"].abs()
    product_mass_flow_HCl = df["Flow Mass Product[bpmed_HCl]"]
    product_mass_flow_NaOH = df["Flow Mass Product[bpmed_NaOH]"]
    # both these variables in kg/s, so still need to multiply by 31557600 s/yr to get kg/yr
    # using mass flow instead of volume flow will give $/kg product instead of $/m^3 product water

    util_factor = df["Utilization Factor"]        #total product water flow, m^3/s, representing denom of "flow"


    if product == "HCl":
        product_flow = df["Flow Mass Product[bpmed_HCl]"]
    else:
        product_flow = df["Flow Mass Product[bpmed_NaOH]"]

    annual_product = product_flow * 31557600.0 * util_factor  # kg product / yr

 # Annualized capex intensity: CapEx share of LCOP
    df["capex_intensity"] = total_capex * crf / annual_product
# Opex intensity: (the OpEx share of LCOP)
    df["opex_intensity"] = total_opex / annual_product
    df["LCOP"] = df["capex_intensity"] + df["opex_intensity"]


# membrane area also

    cell_length = df["Cell Length"]
    cell_width = df["Cell Width"]
    cell_triplet_num = df["Cell Triplet Number"]
    df["membrane area"] = cell_length * cell_width * cell_triplet_num
 
    return df
 
 
def plot_lcop(df, product, feed):

    df = df.sort_values("NaCl Recovery")

    x = df["NaCl Recovery"]
    capex = df["capex_intensity"]
    opex = df["opex_intensity"]
    total = df["LCOP"]
    voltage_applied = df["Voltage Applied"]

    # Calculate bar width based on average recovery step size
    if len(x) > 1:
        step = (x.max() - x.min()) / (len(x) - 1)
        bar_width = step * 0.7  # 70% width leaves 30% gap between bars
    else:
        bar_width = 0.01

    # 3. Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    ax2 = ax.twinx()

    # Base layer: CapEx
    ax.bar(
        x,
        capex,
        width=bar_width,
        label="CapEx Contribution",
        color="#2b5c8f",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )

    # Top layer: OpEx stacked on top of CapEx
    ax.bar(
        x,
        opex,
        width=bar_width,
        bottom=capex,
        label="OpEx Contribution",
        color="#d95f02",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )

    # 4. Optional: Add total LCOP trendline across the top of the bars
    # total_lcop = capex + opex
    # ax.plot(
    #     x,
    #     total,
    #     color="black",
    #     marker="o",
    #     linewidth=1.5,
    #     markersize=4,
    #     label="Total LCOP",
    # )

# voltage on second y axis
    ax2.plot(
        x,
        voltage_applied,
        color = "black",
        linewidth = 2,
        label="Applied Voltage",
    )

    # 5. Formatting
    ax.set_xlabel("NaCl Recovery (-)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel(f"LCOP ($/kg {product})", fontsize=12, fontweight="bold", labelpad=10)
    ax2.set_ylabel(
    "Applied Voltage (V)",
    fontsize=12,
    fontweight="bold",
#    color="red",
    )
    ax2.tick_params(axis="y", colors="black")
    ax2.set_ylim(
    voltage_applied.min() * 0,
    voltage_applied.max() * 1.05,
)

    ax.set_title(
        f"Levelized Cost of {product} Product vs. NaCl Recovery, {feed} g/L feed",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        fontsize=11,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.9),
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust y-axis to leave head-room for legend
#    valid_max = total_lcop.dropna().max() if not total_lcop.dropna().empty else 1.0
#    ax.set_ylim(0, valid_max * 1.2)

    plt.tight_layout()
    return fig
 
 
if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    cases = [
        (70, h5_file_70gL),
        (150, h5_file_150gL),
    ]

    for feed, filename in cases:
        df = load_h5_results(filename)

        # HCl:
        df_hcl = add_normalized_costs(df, product = "HCl")
        fig = plot_lcop(df_hcl, "HCl", feed)

        fig.savefig(
            f"LCOP_HCl_{feed}gL_plot_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )

        #NaOH:
        df_naoh = add_normalized_costs(df, product = "NaOH")
        fig = plot_lcop(df_naoh, "NaOH", feed)

        fig.savefig(
            f"LCOP_NaOH_{feed}gL_plot_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )        
 

    plt.show()
 