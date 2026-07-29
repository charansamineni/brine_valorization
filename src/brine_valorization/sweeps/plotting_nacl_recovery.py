from brine_valorization.costing.valorization_costing_block import (
    ValorizationCostingBlock,
)
import pytest
from brine_valorization.unit_models.bpmed import (
    BPMED,
)

import matplotlib.pyplot as plt

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

    m.fs.bpmed.set_optimization_operation()
 #   m.fs.bpmed.bpmed[0].electrical_stage_num.unfix() 
      # idaes model statistics --> can check electrical_stage_num to see if it's fixed or unfixed

    return m

def run_sweep(nacl_feed, recovery_vals):
    
    solver = get_solver()
    solver.options['max_iter'] = 5000
    solver.options['tol'] = 1e-6
    solver.options['mu_strategy'] = 'adaptive'

    m = build_and_initialize(nacl_feed * pyunits.g / pyunits.L)

    r_start = recovery_vals[0]
    r_stop  = recovery_vals[-1]

    m.fs.bpmed.nacl_recovery.fix(0.7)
    res_warmup = solver.solve(m, tee=False)
    if res_warmup.solver.termination_condition != TerminationCondition.optimal:
        print("Initial 0.70 warm-up solve did not reach optimal termination.")

# adding smooth ramp down
    r_start = recovery_vals[0]
    if abs(0.7 - r_start) > 0.02:
        # Takes 7 small steps from 0.70 down to r_start
        ramp_steps = np.linspace(0.7, r_start, 7)
        for r_ramp in ramp_steps:
            m.fs.bpmed.nacl_recovery.fix(round(r_ramp, 4))
            res_ramp = solver.solve(m, tee=False)
            if res_ramp.solver.termination_condition != TerminationCondition.optimal:
                print(f"Warning: Ramping step at r = {r_ramp:.4f} resulted in {res_ramp.solver.termination_condition}")

    m.fs.bpmed.nacl_recovery.unfix()

    variables_results_list = []

    for r in recovery_vals:
        m.fs.bpmed.nacl_recovery.fix(round(r, 4)) # since recov vals might be weird because of dividing into point- round
        result = solver.solve(m, tee=False)

        if result.solver.termination_condition != TerminationCondition.optimal:
            print(f"Warning: Model failed to converge at NaCl Recovery = {r:.4f}. Skipping.")
            continue

        current_run_data = {
            "NaCl_Recovery": r,
            "Cell_Width_m": value(m.fs.bpmed.bpmed[0].cell_width),
            "Cell_Length_m": value(m.fs.bpmed.bpmed[0].cell_length),
            "Number_of_Triplets": value(m.fs.bpmed.bpmed[0].cell_triplet_num),
            "Base_aem_Outlet_Conc": value(m.fs.bpmed.bpmed[0].salt_conc_aem_x[0, 1]),
            "Acid_cem_Outlet_Conc": value(m.fs.bpmed.bpmed[0].salt_conc_cem_x[0, 1]),
            "Diluate_Outlet_Concentration": value(m.fs.bpmed.bpmed[0].salt_conc_dilu_x[0, 1]),
            "Voltage_Applied": value(m.fs.bpmed.bpmed[0].voltage_applied[0])
        }

        loc_vals = [0, 0.5, 1]
        for loc in loc_vals:
            current_run_data[f"Vel_diluate_{loc}"] = value(m.fs.bpmed.bpmed[0].velocity_diluate[0, loc])
            current_run_data[f"Vel_acidate_{loc}"] = value(m.fs.bpmed.bpmed[0].velocity_acidate[0, loc])
            current_run_data[f"Vel_basate_{loc}"]  = value(m.fs.bpmed.bpmed[0].velocity_basate[0, loc])
            current_run_data[f"Base_aem_Conc_{loc}"] = value(m.fs.bpmed.bpmed[0].salt_conc_aem_x[0, loc])
            current_run_data[f"Acid_cem_Conc_{loc}"] = value(m.fs.bpmed.bpmed[0].salt_conc_cem_x[0, loc])
            current_run_data[f"Diluate_Conc_{loc}"] = value(m.fs.bpmed.bpmed[0].salt_conc_dilu_x[0, loc])

        variables_results_list.append(current_run_data)

    return pd.DataFrame(variables_results_list)


def plot_results(df_results, nacl_feed, timestamp):

    
### PLOTTING ###


## First for Velocities

    plt.figure(figsize=(6,4))

# diluate velocity (for inlet, middle, and outlet)
    plt.plot(df_results["NaCl_Recovery"], df_results["Vel_diluate_0"], label="Inlet (0)", color='gold', marker='o', linestyle='--')
    plt.plot(df_results["NaCl_Recovery"], df_results["Vel_diluate_0.5"], label="Middle (0.5)", color='orange', marker='o', linestyle='--')
    plt.plot(df_results["NaCl_Recovery"], df_results["Vel_diluate_1"], label="Final (1)", color='darkorange', marker='o', linestyle='--')

# acidate velocity (for inlet, middle, and outlet)
    plt.plot(df_results["NaCl_Recovery"], df_results["Vel_acidate_0"], label="Inlet (0)", color='lightcoral', marker='s', linestyle='--')
    plt.plot(df_results["NaCl_Recovery"], df_results["Vel_acidate_0.5"], label="Middle (0.5)", color='red', marker='s', linestyle='--')
    plt.plot(df_results["NaCl_Recovery"], df_results["Vel_acidate_1"], label="Final (1)", color='firebrick', marker='s', linestyle='--')

# basate velocity (for inlet, middle, and outlet)
    plt.plot(df_results["NaCl_Recovery"], df_results["Vel_basate_0"], label="Inlet (0)", color='cornflowerblue', marker='^', linestyle='--')
    plt.plot(df_results["NaCl_Recovery"], df_results["Vel_basate_0.5"], label="Middle (0.5)", color='royalblue', marker='^', linestyle='--')
    plt.plot(df_results["NaCl_Recovery"], df_results["Vel_basate_1"], label="Final (1)", color='blue', marker='^', linestyle='--')

    plt.xlabel("NaCl recovery (fraction)")
    plt.ylabel("Channel Velocities (m/s)")
    plt.title("Velocities through Channel Length vs NaCl Recovery Fraction")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"velocity_vs_recovery_{nacl_feed}_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.show()

## Also for voltage!!

    plt.figure(figsize=(6,4))

# voltage_applied 
    plt.plot(df_results["NaCl_Recovery"], df_results["Voltage_Applied"], color='red', marker='o', linestyle='--')


    plt.xlabel("NaCl recovery (fraction)")
    plt.ylabel("Voltage Applied Across Stack (V)")
    plt.title("Applied Voltage vs NaCl Recovery Fraction")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"voltage_vs_recovery_{nacl_feed}_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Defining sweep setup for each feed concentration (70 g/L and 150 g/L)
    # creating "fine" and "coarse" versions of recovery vals for 150 g/L, because it becomes more sensitive as it approaches limit of 0.74(7)

    r_150_coarse = np.linspace(0.50, 0.65, 8)    # Step ~0.021
    r_150_fine   = np.linspace(0.66, 0.74, 17)   # Step ~0.005
    r_150_array  = np.concatenate([r_150_coarse, r_150_fine])

    sweep_cases = [
            {
                "feed": 70, "recovery_vals": np.linspace(0.50, 0.82, 17)
            },
            # {
            #     "feed": 150, "recovery_vals": r_150_array
            # }
        ]

    for case in sweep_cases:
        feed = case["feed"]
        recovery_vals = case["recovery_vals"]
        
        df_results = run_sweep(
            nacl_feed=feed, 
            recovery_vals=recovery_vals
        )

    # save everything- by time stamp so it doesn't overwrite previous versions!
        csv_filename = f"nacl_recovery_sweep_{feed}gL_{timestamp}.csv"
        df_results.to_csv(csv_filename, index=False)


        #Print first and last point's conditions (for lowest and highest recoveries)
        print(f"\n--- Key Variables at First Point (Lowest Recovery, {feed} g/L) ---")
        print(df_results.iloc[0])
        print(f"\n--- Key Variables at Last Point (Highest Recovery, {feed} g/L) ---")
        print(df_results.iloc[-1])

       # PLOTTING FUNCTION! (and saves plots)
        plot_results(df_results, nacl_feed=feed, timestamp=timestamp)



if __name__ == "__main__":

    main()

