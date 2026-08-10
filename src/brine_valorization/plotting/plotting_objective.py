import pandas as pd
import h5py
import matplotlib.pyplot as plt
from datetime import datetime

from sympy import product

# m.fs.product_objective = Objective(
#     expr=m.fs.costing.LCOP
#     + sum(
#         (1 - m.fs.bpmed.product_mass_concentration[p])
#         for p in m.fs.bpmed.product_mass_concentration
#     )
# )


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

            "Voltage Applied":
                f["outputs/fs.bpmed.bpmed[0].voltage_applied[0.0]/value"] [:],
            "Cell Length":
                f["outputs/fs.bpmed.bpmed[0].cell_length/value"] [:],
            "Cell Width":
                f["outputs/fs.bpmed.bpmed[0].cell_width/value"] [:],
            "Cell Triplet Number":
                f["outputs/fs.bpmed.bpmed[0].cell_triplet_num/value"] [:],

            "Product Objective": f["outputs/fs.product_objective/value"][:],
            "LCOP": f["outputs/fs.costing.LCOP/value"][:],
            "Product Mass Concentration HCl": f["outputs/fs.bpmed.product_mass_concentration[bpmed,HCl]/value"][:],
            "Product Mass Concentration NaOH": f["outputs/fs.bpmed.product_mass_concentration[bpmed,NaOH]/value"][:],

            
        })

    return df
 

 
def plot_objective(df, product, feed):

    df = df.sort_values("NaCl Recovery")

    x = df["NaCl Recovery"]
    # capex = df["capex_intensity"]
    # opex = df["opex_intensity"]
    # total = df["LCOP"]
    # voltage_applied = df["Voltage Applied"]
    # membrane_area = df["membrane area"]
    product_objective = df["Product Objective"]
    product_mass_con_HCl = df["Product Mass Concentration HCl"]
    product_mass_con_NaOH = df["Product Mass Concentration NaOH"]
    lcop = df["LCOP"]
    df["concentration_objective_component"] = ((1 - product_mass_con_HCl) + (1 - product_mass_con_NaOH))
    conc_obj= df["concentration_objective_component"]


    fig, ax = plt.subplots(figsize=(10, 6))

    #ax2 = ax.twinx()


  
    ax.plot(
        x,
        product_objective,
        color="black",
        marker="o",
        linewidth=1.5,
        markersize=4,
        label="Total Objective",
    )

 
    # plotting LCOP on primary y-axis
    ax.plot(
        x,
        lcop,
        color="green",
        linewidth=2,
        label="LCOP",
    )
    # y_label = "LCOP ($/kg product)"
    y_label = "Product Objective"
    legend_position = (0.0, 0.9)


        # voltage on primary y axis
    #ax2.plot(
    ax.plot(
        x,
        conc_obj,
        color = "orange",
        linewidth = 2,
        label="Concentration Objective Component",
    )
    y_label_2 = "Product Concentration Component"
    # ax.set_ylim(
    #     voltage_applied.min() * 0,
    #     voltage_applied.max() * 1.05,
    # )




    # 5. Formatting
    ax.set_xlabel("NaCl Recovery (-)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold", labelpad=10)
#     ax2.set_ylabel(
#     y_label_2,
#     fontsize=12,
#     fontweight="bold",
# #    color="red",
#     )
#     ax2.tick_params(axis="y", colors="black")


    ax.set_title(
        f"Product Objective vs. NaCl Recovery, {feed} g/L feed",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # handles1, labels1 = ax.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(
        # handles1 + handles2,
        # labels1 + labels2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        fontsize=11,
        loc="upper center",
    #    bbox_to_anchor=legend_position,
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
        # df_hcl = add_normalized_costs(df, product = "HCl")
        fig = plot_objective(df, "HCl", feed)

        fig.savefig(
            f"Objective_sameAxis{feed}gL_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )

        # #NaOH:
        # df_naoh = add_normalized_costs(df, product = "NaOH")
        # fig = plot_objective(df_naoh, "NaOH", feed)

        # fig.savefig(
        #     f"LCOP_ConcentrationObjective_{feed}gL_{timestamp}.png",
        #     dpi=300,
        #     bbox_inches="tight",
        # )        
 

    plt.show()
 