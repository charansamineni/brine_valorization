import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# plot from costing_sweep.py 's parameter sweep code!
costing_vars = [
    "fs.costing.total_investment_factor",
    "fs.costing.maintenance_labor_chemical_factor",
    "fs.costing.utilization_factor",
    "fs.costing.electricity_cost",
    "fs.costing.electrical_carbon_intensity",
    "fs.costing.plant_lifetime",
    "fs.costing.wacc",
    "fs.costing.capital_recovery_factor",
    "fs.costing.TPEC",
    "fs.costing.TIC",
    "fs.costing.aggregate_capital_cost",
    "fs.costing.aggregate_fixed_operating_cost",
    "fs.costing.aggregate_variable_operating_cost",
    "fs.costing.aggregate_flow_electricity",
    "fs.costing.aggregate_flow_costs[electricity]",
    "fs.costing.aggregate_direct_capital_cost",
    "fs.costing.total_capital_cost",
    "fs.costing.total_operating_cost",
    "fs.costing.maintenance_labor_chemical_operating_cost",
    "fs.costing.total_fixed_operating_cost",
    "fs.costing.total_variable_operating_cost",
    "fs.costing.total_annualized_cost",
    "fs.costing.annual_product_generation",
    "fs.costing.LCOP",
    "fs.costing.specific_energy_consumption",
    "fs.costing.bipolar_electrodialysis_costing.membrane_capital_cost",
    "fs.costing.bipolar_electrodialysis_costing.factor_membrane_replacement",
    "fs.costing.bipolar_electrodialysis_costing.stack_electrode_capital_cost",
    "fs.costing.bipolar_electrodialysis_costing.factor_stack_electrode_replacement",
    "fs.costing.low_pressure_pump.unit_cost",
]

#how LCOW is calculated
# fs.costing.total_capital_cost*fs.costing.capital_recovery_factor 
# + fs.costing.total_operating_cost) / (31557600.000000004*(s/a)*fs.total_product_water*fs.costing.utilization_factor)


CSV_PATH = "updated_costing_vs_recovery_150gL_2026-08-04_1610.csv"  # <-- put your CSV filename here

df = pd.read_csv(CSV_PATH)

# Remove duplicate column names, keeping the first one
df = df.loc[:, ~df.columns.duplicated()]

df.to_csv(CSV_PATH, index=False)


RECOVERY_COL = "NaCl Recovery"   # <-- point this at your actual recovery column
# MASS_FLOW_COL = None             # <-- optional: kg/hr product flow column, if saved
 
 
def add_normalized_costs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    crf = df["Capital Recovery Factor"]
    total_capex = df["Total Capital Cost"]      # $ (at fs level)
    total_opex = df["Total Operating Cost"]      # $/yr, total fixed opex + total variable opex (at fs level)
    total_prod_water = df["Total Product Water"].abs()
    util_factor = df["Utilization Factor"]        #total product water flow, m^3/s, representing denom of "flow"

    #annual_product = df["fs.costing.annual_product_generation"]  # kg product / yr
 

    # if MASS_FLOW_COL is not None:
    #     df["specific_capital_cost"] = total_capex / df[MASS_FLOW_COL]
 
    # Annualized capex intensity: CapEx share of LCOW
    df["capex_intensity"] = (total_capex * crf) / (total_prod_water * 31557600.0 * util_factor )  # convert to $/kg product
 
    # Opex intensity: (the OpEx share of LCOW)
    df["opex_intensity"] = total_opex / (total_prod_water * 31557600.0 * util_factor )
 
    # Sanity check: these two should sum to fs.costing.LCOW
    df["LCOW_check"] = df["capex_intensity"] + df["opex_intensity"]
 
    return df
 
 
def plot_stacked_lcow_vs_recovery(df: pd.DataFrame, recovery_col="NaCl Recovery"):
    """Plots LCOW vs NaCl Recovery as a stacked bar chart (CapEx + OpEx).

    Parameters:
        df: DataFrame containing the sweep data.
        recovery_col: Column name for NaCl recovery (e.g., 'NaCl Recovery' or
          'fs.bpmed.nacl_recovery').
    """
    # 1. Apply cost normalization if not already run
    if "capex_intensity" not in df.columns or "opex_intensity" not in df.columns:
        df = add_normalized_costs(df)

    # 2. Sort by recovery rate for clean plotting
    df = df.sort_values(by=recovery_col).reset_index(drop=True)

    x = df[recovery_col]
    capex = df["capex_intensity"]
    opex = df["opex_intensity"]

    # Calculate bar width based on average recovery step size
    if len(x) > 1:
        step = (x.max() - x.min()) / (len(x) - 1)
        bar_width = step * 0.7  # 70% width leaves 30% gap between bars
    else:
        bar_width = 0.01

    # 3. Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

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

    # 4. Optional: Add total LCOW trendline across the top of the bars
    total_lcow = capex + opex
    ax.plot(
        x,
        total_lcow,
        color="black",
        marker="o",
        linewidth=1.5,
        markersize=4,
        label="Total LCOW",
    )

    # 5. Formatting
    ax.set_xlabel("NaCl Recovery (-)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("LCOW ($/m³ product water)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(
        "Levelized Cost of Water Breakdown (CapEx, OpEx) vs. NaCl Recovery (150 g/L feed)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=11, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust y-axis to leave head-room for legend
    valid_max = total_lcow.dropna().max() if not total_lcow.dropna().empty else 1.0
    ax.set_ylim(0, valid_max * 1.2)

    plt.tight_layout()
    return fig
 
 
if __name__ == "__main__":
    print(f"Loading {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, comment=None)

    df.columns = df.columns.str.lstrip("# ").str.strip()

    df = df.loc[:, ~df.columns.duplicated(keep="first")]
 
    # missing = [c for c in (RECOVERY_COL, "fs.costing.total_capital_cost") if c not in df.columns]
    # if missing:
    #     raise KeyError(
    #         f"Column(s) not found in {CSV_PATH}: {missing}. "
    #         f"Available columns:\n{list(df.columns)}"
    #     )
 
    df = add_normalized_costs(df)
    fig = plot_stacked_lcow_vs_recovery(df, recovery_col=RECOVERY_COL)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = f"capex_opex_vs_recovery_150gL_{timestamp}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

    plt.show()
 
    # LCOW sanity check
    max_err = (df["LCOW_check"] - df["fs.costing.LCOW"]).abs().max()
    print(f"Max |capex_intensity + opex_intensity - LCOW| = {max_err:.6g}")