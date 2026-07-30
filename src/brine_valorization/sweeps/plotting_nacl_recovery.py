from brine_valorization.costing.valorization_costing_block import (
    ValorizationCostingBlock,
)
import pytest
from brine_valorization.unit_models.bpmed import (
    BPMED,
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from pyomo.environ import (
    TransformationFactory,
    assert_optimal_termination,
)
from pyomo.opt import TerminationCondition

import idaes.core.util.scaling as iscale
from idaes.core.util.exceptions import InitializationError

from pyomo.environ import (
    Var,
    value,
    Constraint,
    Objective,
    ConcreteModel,
    units as pyunits,
)

from idaes.core import (
    FlowsheetBlock,
)
from reaktoro_enabled_watertap.unit_models.multi_comp_feed_unit import (
    MultiCompFeed,
)
from brine_valorization.property_models.mcas_with_enthalpy import (
    MCASWEParameterBlock,
)
from watertap.property_models.multicomp_aq_sol_prop_pack import (
    ActivityCoefficientModel,
    DensityCalculation,
)
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.diagnostics_tools.diagnostics_toolbox import DiagnosticsToolbox
import sys
import numpy as np
import pandas as pd
from datetime import datetime

try:
    # needed for windows dealing with ohm symbol encoding
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
except:
    pass

from watertap.property_models.water_prop_pack import WaterParameterBlock
from watertap.core.solvers import get_solver


def build_BPMED_feed_cases(NaCl=150 * pyunits.g / pyunits.L):
    mols_nacl = NaCl / (22.98977 + 35.45)
    mass_conc_na = mols_nacl * 22.98977
    mass_conc_cl = mols_nacl * 35.45
    # Dilute streams are 100x less concentrated
    mols_nacl_dilute = (NaCl/100) / (22.98977 + 35.45)
    mass_conc_na_dilute = mols_nacl_dilute * 22.98977
    mass_conc_cl_dilute = mols_nacl_dilute * 35.45
    mcas_props = {
        "solute_list": ["Na_+", "Cl_-"],
        "mw_data": {"H2O": 18e-3, "Na_+": 22.98977e-3, "Cl_-": 35.45e-3},
        "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8},
        "charge": {"Na_+": 1, "Cl_-": -1},
        "diffusivity_data": {("Liq", "Na_+"): 1.33e-9, ("Liq", "Cl_-"): 2.03e-9},
        "activity_coefficient_model": ActivityCoefficientModel.ideal,
        "density_calculation": DensityCalculation.constant,
    }
    m = ConcreteModel()
    m.fs = FlowsheetBlock()
    m.fs.properties = MCASWEParameterBlock(**mcas_props)
    m.fs.dilute_properties = MCASWEParameterBlock(**mcas_props)
    m.fs.feed = MultiCompFeed(
        default_property_package=m.fs.properties,
        reconcile_using_reaktoro=False,
        temperature=298.15,
        pressure=101325,
        volumetric_flowrate=1 * pyunits.L / pyunits.s,
        ion_concentrations={"Na_+": mass_conc_na, "Cl_-": mass_conc_cl},
    )
    m.fs.dilute_feed = MultiCompFeed(
        default_property_package=m.fs.dilute_properties,
        reconcile_using_reaktoro=False,
        temperature=298.15,
        pressure=101325,
        volumetric_flowrate=1 * pyunits.L / pyunits.s,
        ion_concentrations={"Na_+": mass_conc_na_dilute, "Cl_-": mass_conc_cl_dilute},
    )
    m.fs.feed.fix_and_scale()
    m.fs.feed.report()
    return m

def add_costing():

    return 


def build_and_initialize(NaCl):
    """Build a fresh model for a given feed concentration, initialize it,
    and leave it ready for the optimization sweep (set_optimization_operation
    already called, nacl_recovery already unfixed by that call)."""
    m = build_BPMED_feed_cases(NaCl=NaCl)

    m.fs.costing = ValorizationCostingBlock()
    m.fs.bpmed = BPMED(
        default_property_package=m.fs.properties,
        default_costing_package=m.fs.costing,
    )
    m.fs.feed.fix_and_scale()
    m.fs.dilute_feed.fix_and_scale()

    m.fs.bpmed.fix_and_scale()
    m.fs.feed.outlet.connect_to(m.fs.bpmed.brine_inlet)
    m.fs.dilute_feed.outlet.connect_to(m.fs.bpmed.low_tds_water_inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

    m.fs.costing.cost_process()
    m.fs.costing.add_annual_product_generation(
        sum(m.fs.bpmed.flow_mass_product[p] for p in m.fs.bpmed.flow_mass_product)
    )
    m.fs.costing.add_LCOP(
        sum(m.fs.bpmed.flow_mass_product[p] for p in m.fs.bpmed.flow_mass_product)
    )
    m.fs.costing.add_mass_based_specific_energy_consumption(
        sum(m.fs.bpmed.flow_mass_product[p] for p in m.fs.bpmed.flow_mass_product)
    )

    m.fs.costing.initialize()
    m.fs.product_objective = Objective(
        expr=m.fs.costing.LCOP
        + sum(
            (1 - m.fs.bpmed.product_mass_concentration[p])
            for p in m.fs.bpmed.product_mass_concentration
        )
    )

    iscale.calculate_scaling_factors(m)
    assert degrees_of_freedom(m) == 0

    m.fs.feed.initialize()
    m.fs.dilute_feed.initialize()


    m.fs.bpmed.initialize()

    assert degrees_of_freedom(m) == 0

    solver = get_solver()
    result = solver.solve(m, tee=True)
#    assert_optimal_termination(result)

    m.fs.bpmed.set_optimization_operation()
 #   m.fs.bpmed.bpmed[0].electrical_stage_num.unfix() 
      # idaes model statistics --> can check electrical_stage_num to see if it's fixed or unfixed

    return m

def run_sweep(nacl_feed, recovery_vals):
    
    solver = get_solver()
    solver.options['max_iter'] = 5000
    solver.options['tol'] = 1e-5
    solver.options['mu_strategy'] = 'adaptive'      # this let it solve!! 

    m = build_and_initialize(nacl_feed * pyunits.g / pyunits.L)

    r_start = recovery_vals[0]

    m.fs.bpmed.nacl_recovery.fix(round(r_start, 4))
    res_warmup = solver.solve(m, tee=False)
 #   assert_optimal_termination(res_warmup)
    if res_warmup.solver.termination_condition != TerminationCondition.optimal:
        print(f"Warning: Initial warm-up solve at r = {r_start:.4f} resulted in {res_warmup.solver.termination_condition}")

# adding smooth ramp down
    # r_start = recovery_vals[0]
    # if abs(0.7 - r_start) > 0.02:
    #     # Takes 7 small steps from 0.70 down to r_start
    #     ramp_steps = np.linspace(0.7, r_start, 7)
    #     for r_ramp in ramp_steps:
    #         m.fs.bpmed.nacl_recovery.fix(round(r_ramp, 4))
    #         res_ramp = solver.solve(m, tee=False)
    #         #assert_optimal_termination(res_ramp)
    #         if res_ramp.solver.termination_condition != TerminationCondition.optimal:
    #             print(f"Warning: Ramping step at r = {r_ramp:.4f} resulted in {res_ramp.solver.termination_condition}")

    m.fs.bpmed.nacl_recovery.unfix()

    all_var_records = []
    key_variables_results = []

    for r in recovery_vals:
        m.fs.bpmed.nacl_recovery.fix(round(r, 4)) # since recov vals might be weird because of dividing into point- round
        result = solver.solve(m, tee=False)
#        assert_optimal_termination(result)

        if result.solver.termination_condition != TerminationCondition.optimal:
            print(f"Warning: Model failed to converge at NaCl Recovery = {r:.4f}. Skipping.")
            continue

        for v in m.component_data_objects(Var, active=True):
            all_var_records.append({
                "NaCl_Recovery": round(r, 4),
                "Variable_Name": v.name,
                "Value": value(v, exception=False),
                "Lower_Bound": v.lb if v.lb is not None else -np.inf,
                "Upper_Bound": v.ub if v.ub is not None else np.inf,
                "Is_Fixed": v.fixed,
            })

        current_run_data = {
            "NaCl_Recovery": r,
            "Cell_Width_m": value(m.fs.bpmed.bpmed[0].cell_width),
            "Cell_Length_m": value(m.fs.bpmed.bpmed[0].cell_length),
            "Number_of_Triplets": value(m.fs.bpmed.bpmed[0].cell_triplet_num),
            "Base_aem_Outlet_Conc": value(m.fs.bpmed.bpmed[0].salt_conc_aem_x[0, 1]),
            "Acid_cem_Outlet_Conc": value(m.fs.bpmed.bpmed[0].salt_conc_cem_x[0, 1]),
            "Diluate_Outlet_Concentration": value(m.fs.bpmed.bpmed[0].salt_conc_dilu_x[0, 1]),
            "Voltage_Applied": value(m.fs.bpmed.bpmed[0].voltage_applied[0]),
            "Areal_Resistance": value(m.fs.bpmed.bpmed[0].total_areal_resistance_x[0, 0]),
            "LimBPEM_Current_Density": value(m.fs.bpmed.bpmed[0].current_dens_lim_bpem[0, 1]),
            "LCOP": value(m.fs.costing.LCOP),
            "SEC": value(m.fs.costing.specific_energy_consumption)

        }

        # loc_vals = [0, 0.5, 1]
        # for loc in loc_vals:
        #     current_run_data[f"Vel_diluate_{loc}"] = value(m.fs.bpmed.bpmed[0].velocity_diluate[0, loc])
        #     current_run_data[f"Vel_acidate_{loc}"] = value(m.fs.bpmed.bpmed[0].velocity_acidate[0, loc])
        #     current_run_data[f"Vel_basate_{loc}"]  = value(m.fs.bpmed.bpmed[0].velocity_basate[0, loc])
        #     current_run_data[f"Base_aem_Conc_{loc}"] = value(m.fs.bpmed.bpmed[0].salt_conc_aem_x[0, loc])
        #     current_run_data[f"Acid_cem_Conc_{loc}"] = value(m.fs.bpmed.bpmed[0].salt_conc_cem_x[0, loc])
        #     current_run_data[f"Diluate_Conc_{loc}"] = value(m.fs.bpmed.bpmed[0].salt_conc_dilu_x[0, loc])
         #   channel_location_list.append(value(m.fs.bpmed.bpmed[0].cell_length) * loc)

        loc_vals = sorted(set(k[1] for k in m.fs.bpmed.bpmed[0].velocity_diluate.keys()))

        for loc in loc_vals:
            # Key formatting (e.g., 0.0 -> "0", 0.1 -> "0.1", 1.0 -> "1")
            loc_str = f"{float(loc):g}"
            current_run_data[f"Vel_diluate_{loc_str}"] = value(m.fs.bpmed.bpmed[0].velocity_diluate[0, loc])
            current_run_data[f"Vel_acidate_{loc_str}"] = value(m.fs.bpmed.bpmed[0].velocity_acidate[0, loc])
            current_run_data[f"Vel_basate_{loc_str}"]  = value(m.fs.bpmed.bpmed[0].velocity_basate[0, loc])


        key_variables_results.append(current_run_data)

    return all_var_records, pd.DataFrame(key_variables_results)


def convert_to_long_format(df_wide):
    """Translates relative channel coordinates (all available discretization nodes)
    into physical channel lengths (m).
    """
    # Auto-discover all velocity locations from column names
    vel_cols = [c for c in df_wide.columns if c.startswith("Vel_diluate_")]

    loc_info = []
    for col in vel_cols:
        loc_str = col.replace("Vel_diluate_", "")
        loc_info.append((loc_str, float(loc_str)))

    # Sort spatially from inlet (0) to outlet (1)
    loc_info.sort(key=lambda x: x[1])

    long_rows = []
    for _, row in df_wide.iterrows():
        rec = row["NaCl_Recovery"]
        cell_len = row["Cell_Length_m"]

        for loc_str, loc_val in loc_info:
            abs_pos = loc_val * cell_len

            for stream in ["diluate", "acidate", "basate"]:
                col_name = f"Vel_{stream}_{loc_str}"
                if col_name in row:
                    long_rows.append({
                        "NaCl_Recovery": rec,
                        "Stream": stream.capitalize(),
                        "Rel_Loc": loc_val,
                        "Cell_Length_m": cell_len,
                        "Abs_Position_m": abs_pos,
                        "Velocity_m_s": row[col_name]
                    })

    return pd.DataFrame(long_rows)

def plot_spatial_velocity_subplots(df_wide, nacl_feed, timestamp):
    """Plots spatial velocity profiles for Diluate, Acidate, and Basate in side-by-side subplots."""
    # Convert data to long format
    df_long = convert_to_long_format(df_wide)
    # Setup figure with 3 subplots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    streams = ["Diluate", "Acidate", "Basate"]
#    stream_markers = {"Diluate": "o", "Acidate": "s", "Basate": "^"}
    velocity_bounds = {
        "Diluate": {"lb": 0.01, "ub": 0.25},
        "Acidate": {"lb": 0.01, "ub": 0.25},
        "Basate": {"lb": 0.01, "ub": 0.25},
    }

    # Color map setup for NaCl Recovery
    recoveries = sorted(df_long["NaCl_Recovery"].unique())
    norm = mcolors.Normalize(vmin=min(recoveries), vmax=max(recoveries))
    cmap = cm.viridis

    for ax, stream in zip(axes, streams):
        df_stream = df_long[df_long["Stream"] == stream]

        for rec in recoveries:
            df_sub = df_stream[df_stream["NaCl_Recovery"] == rec].sort_values("Abs_Position_m")
            color = cmap(norm(rec))

            ax.plot(
                df_sub["Abs_Position_m"], 
                df_sub["Velocity_m_s"],  
                color=color,
                alpha=0.85
            )

        lb = velocity_bounds[stream]["lb"]
        ub = velocity_bounds[stream]["ub"]

        ax.axhline(
            y=lb,
            color="firebrick",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="Velocity Bounds",
        )
        # ax.axhline(
        #     y=ub, color="firebrick", linestyle="--", linewidth=1.2, alpha=0.8
        # )

        ax.set_xlabel("Channel Position (m)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title(f"{stream} Channel", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)

    # Set y-label only on the leftmost plot (since sharey=True)
    axes[0].set_ylabel("Velocity (m/s)")

    # Unified Colorbar for Recovery
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.03)
    cbar.set_label("NaCl Recovery (fraction)", rotation=270, labelpad=15)

    plt.suptitle(f"Channel Velocity Spatial Profiles ({nacl_feed} g/L Feed)", fontsize=14, y=1.02)
    plt.savefig(f"velocity_spatial_subplots_{nacl_feed}_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.show()




def plot_combined_voltage(results_dict, timestamp):

    
### PLOTTING ###
#     plt.figure(figsize=(6,4))

# # diluate velocity (for inlet, middle, and outlet)
#     plt.plot(df_results["NaCl_Recovery"], df_results["Vel_diluate_0"], label="Inlet (0)", color='gold')
#     plt.plot(df_results["NaCl_Recovery"], df_results["Vel_diluate_0.5"], label="Middle (0.5)", color='orange')
#     plt.plot(df_results["NaCl_Recovery"], df_results["Vel_diluate_1"], label="Final (1)", color='darkorange')

# # acidate velocity (for inlet, middle, and outlet)
#     plt.plot(df_results["NaCl_Recovery"], df_results["Vel_acidate_0"], label="Inlet (0)", color='lightcoral')
#     plt.plot(df_results["NaCl_Recovery"], df_results["Vel_acidate_0.5"], label="Middle (0.5)", color='red')
#     plt.plot(df_results["NaCl_Recovery"], df_results["Vel_acidate_1"], label="Final (1)", color='firebrick')

# # basate velocity (for inlet, middle, and outlet)
#     plt.plot(df_results["NaCl_Recovery"], df_results["Vel_basate_0"], label="Inlet (0)", color='cornflowerblue')
#     plt.plot(df_results["NaCl_Recovery"], df_results["Vel_basate_0.5"], label="Middle (0.5)", color='royalblue')
#     plt.plot(df_results["NaCl_Recovery"], df_results["Vel_basate_1"], label="Final (1)", color='blue')

#     legend_elements = [
#     Line2D([0], [0], marker='o', color='darkorange', linestyle='--', label='Diluate', markersize=8),
#     Line2D([0], [0], marker='s', color='red', linestyle='--', label='Acidate', markersize=8),
#     Line2D([0], [0], marker='^', color='blue', linestyle='--', label='Basate', markersize=8),
#     ]

#     plt.legend(handles=legend_elements, loc='upper left', frameon=True)

#     plt.xlabel("NaCl recovery (fraction)")
#     plt.ylabel("Channel Velocities (m/s)")
#     plt.title("Velocities through Channel Length vs NaCl Recovery Fraction")
#     plt.grid(True)
#     plt.tight_layout()

#     plt.savefig(f"velocity_vs_recovery_{nacl_feed}_{timestamp}.png", dpi=300, bbox_inches="tight")
#     plt.show()

## Also for voltage!!

    plt.figure(figsize=(7,5))

    colors = {70: "teal", 150: "darkorange"}

    for feed, df_results in results_dict.items():
        color = colors.get(feed, "blue")
        plt.plot(
            df_results["NaCl_Recovery"],
            df_results["Voltage_Applied"],
            marker='o',
            markersize=4,
            linewidth=1.8,
            color=color,
            label=f"{feed} g/L Feed",
        )

    plt.xlabel("NaCl recovery (fraction)")
    plt.ylabel("Voltage Applied Across Stack (V)")
    plt.title("Applied Voltage vs NaCl Recovery Fraction for 2 Feed Cases")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    plt.savefig(f"voltage_vs_recovery_combined_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

    # Defining sweep setup for each feed concentration (70 g/L and 150 g/L)
    # creating "fine" and "coarse" versions of recovery vals for 150 g/L, because it becomes more sensitive as it approaches limit of 0.74(7)

    # r_150_coarse = np.linspace(0.50, 0.65, 8)    # Step ~0.021
    # r_150_fine   = np.linspace(0.66, 0.74, 17)   # Step ~0.005
    # r_150_array  = np.concatenate([r_150_coarse, r_150_fine])
    r_150_array  = np.linspace(0.5, 0.74, 25)

    sweep_cases = [
            {
                "feed": 70, "recovery_vals": np.linspace(0.50, 0.82, 25)
            },
            {
                "feed": 150, "recovery_vals": r_150_array
            }
        ]

    results_dict = {}

    for case in sweep_cases:
        feed = case["feed"]
        recovery_vals = case["recovery_vals"]
        
        # df_results = run_sweep(
        #     nacl_feed=feed, 
        #     recovery_vals=recovery_vals
        # )
        all_var_records, df_results = run_sweep(nacl_feed=feed, recovery_vals=recovery_vals)
        df_full_model = pd.DataFrame(all_var_records)

        # saving 2 feed cases results to plot combined voltage
        results_dict[feed] = df_results

    # save everything- by time stamp so it doesn't overwrite previous versions!
        # csv_filename = f"nacl_recovery_keyvars_{feed}gL_{timestamp}.csv"
        # df_results.to_csv(csv_filename, index=False)

        h5_filename = f"full_model_all_vars_{feed}gL_{timestamp}.h5"
        df_full_model.to_hdf(h5_filename, key="variables", mode="w", format="table")

        # Compressed CSV format
        full_csv_filename = f"full_model_all_vars_{feed}gL_{timestamp}.csv.gz"
        df_full_model.to_csv(full_csv_filename, index=False, compression="gzip")
        print(f"Saved complete model state to compressed CSV: {full_csv_filename}")


        #Print first and last point's conditions (for lowest and highest recoveries)
        print(f"\n--- Key Variables at First Point (Lowest Recovery, {feed} g/L) ---")
        print(df_results.iloc[0])
        print(f"\n--- Key Variables at Last Point (Highest Recovery, {feed} g/L) ---")
        print(df_results.iloc[-1])

       # Plotting velocity for each feed case (and saves plots)
        plot_spatial_velocity_subplots(df_results, feed, timestamp)

    plot_combined_voltage(results_dict, timestamp=timestamp)


if __name__ == "__main__":

    main()

