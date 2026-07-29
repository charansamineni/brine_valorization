import sys
import numpy as np
import pytest

from pyomo.environ import (
    Var,
    value,
    Objective,
    ConcreteModel,
    units as pyunits,
    TransformationFactory,
)
from pyomo.opt import TerminationCondition

from idaes.core import FlowsheetBlock
import idaes.core.util.scaling as iscale
from idaes.core.util.exceptions import InitializationError
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.model_diagnostics import DiagnosticsToolbox  # Consolidated to the modern import

from watertap.core.solvers import get_solver
from watertap.property_models.multicomp_aq_sol_prop_pack import (
    ActivityCoefficientModel,
    DensityCalculation,
)
from reaktoro_enabled_watertap.unit_models.multi_comp_feed_unit import MultiCompFeed
from brine_valorization.property_models.mcas_with_enthalpy import MCASWEParameterBlock
from brine_valorization.costing.valorization_costing_block import ValorizationCostingBlock
from brine_valorization.unit_models.bpmed import BPMED

# Handle Windows ohm symbol encoding issues safely
try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

def build_BPMED_feed_cases(NaCl_conc=150 * pyunits.g / pyunits.L):
    # Calculations
    mols_nacl = NaCl_conc / (22.98977 + 35.45)
    mass_conc_na = mols_nacl * 22.98977
    mass_conc_cl = mols_nacl * 35.45
    
    # Dilute streams are 100x less concentrated
    mols_nacl_dilute = mols_nacl / 100
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
    return m

def build_and_initialize(NaCl_conc):
    m = build_BPMED_feed_cases(NaCl_conc=NaCl_conc)

    m.fs.costing = ValorizationCostingBlock()
    m.fs.bpmed = BPMED(
        default_property_package=m.fs.properties,
        default_costing_package=m.fs.costing,
    )
    
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

    try:
        m.fs.bpmed.initialize()
    except InitializationError as e:
        print(f"\n*** m.fs.bpmed.initialize() failed: {e} ***\n")
        dt = DiagnosticsToolbox(m)
        
        print("\n--- 1. Variables causing the NoneType crash ---")
        try:
            dt.display_variables_with_none_value_in_activated_constraints()
        except Exception as none_err:
            print(f"Could not display None variables: {none_err}")

        # Force-initialize any None variables so the rest of the diagnostics can run
        for v in m.component_data_objects(Var, active=True):
            if v.value is None:
                v.set_value(1e-8)  # small non-zero dummy value

        print("\n--- 2. Variables at or outside bounds ---")
        dt.display_variables_at_or_outside_bounds()
        
        print("\n--- 3. Constraints with large residuals ---")
        dt.display_constraints_with_large_residuals()

        print("\n--- 4. Numerical Issues Report ---")
        try:
            dt.report_numerical_issues()
        except Exception as dt_err:
            print(f"Could not run numerical issues report: {dt_err}")
            
        raise

    m.fs.costing.initialize()
    assert degrees_of_freedom(m) == 0

    solver = get_solver()
    print("Solving fixed-operation model...")
    result = solver.solve(m, tee=True)
    print(result.solver.termination_condition)

    m.fs.bpmed.set_optimization_operation()
    return m

def find_recovery_limit(m, r_start=0.6, r_stop=0.5, coarse_step=0.02, fine_step=0.002, scan_max_iter=7000):
    solver = get_solver()
    solver.options['max_iter'] = scan_max_iter
    solver.options['tol'] = 1e-6

    last_good_r = r_start
    first_fail_r = None

    def try_solve(r):
        m.fs.bpmed.nacl_recovery.fix(round(r, 4))
        print(f"\n--- Trying nacl_recovery = {r:.4f} ---")
        try:
            result = solver.solve(m, tee=False)
            ok = result.solver.termination_condition == TerminationCondition.optimal
        except Exception as e:
            print(f"  -> EXCEPTION: {e}")
            return False
        print(f"  -> {'converged' if ok else 'NOT optimal: ' + str(result.solver.termination_condition)}")
        return ok

    # Coarse ramp
    r = r_start
    while r > r_stop:
        r_next = round(r - coarse_step, 4)
        if try_solve(r_next):
            last_good_r = r_next
            r = r_next
        else:
            first_fail_r = r_next
            break
    else:
        print("Reached r_stop without failure.")
        return last_good_r, None

    # Fine bisection
    hi, lo = last_good_r, first_fail_r
    while hi - lo > fine_step:
        mid = round((lo + hi) / 2, 4)
        if try_solve(mid):
            last_good_r = hi = mid
        else:
            first_fail_r = lo = mid

    print(f"\nLast successful recovery: {last_good_r}")
    print(f"First failing recovery:   {first_fail_r}")

    # FIX: Wrap the final diagnostics solve in a try-except so it doesn't crash the script
    m.fs.bpmed.nacl_recovery.fix(first_fail_r)
    try:
        solver.solve(m, tee=False) 
    except Exception as e:
        print(f"\n[Warning] Final solve for failed iterate threw an exception (expected). Error: {e}")
        
    return last_good_r, first_fail_r

def print_suspect_bounds(m):
    b = m.fs.bpmed
    candidates = [
        ("low_tds_splitter.split_fraction[acidate]", getattr(b, "low_tds_splitter", None) and b.low_tds_splitter.split_fraction[0, "acidate"]),
        ("acidate_splitter.split_fraction[recycle]", getattr(b, "acidate_splitter", None) and b.acidate_splitter.split_fraction[0, "recycle"]),
        ("basate_splitter.split_fraction[recycle]", getattr(b, "basate_splitter", None) and b.basate_splitter.split_fraction[0, "recycle"]),
        ("voltage_applied", getattr(b.bpmed[0], "voltage_applied", None) and b.bpmed[0].voltage_applied[0]),
        ("cell_length", getattr(b.bpmed[0], "cell_length", None)),
        ("cell_triplet_num", getattr(b.bpmed[0], "cell_triplet_num", None)),
        ("velocity_diluate", getattr(b.bpmed[0], "velocity_diluate", None) and b.bpmed[0].velocity_diluate[0, 0]),
        ("velocity_acidate", getattr(b.bpmed[0], "velocity_acidate", None) and b.bpmed[0].velocity_acidate[0, 0]),
        ("velocity_basate", getattr(b.bpmed[0], "velocity_basate", None) and b.bpmed[0].velocity_basate[0, 0]),
    ]
    print(f"\n{'Variable':40s} {'Value':>12s} {'LB':>10s} {'UB':>10s}")
    for name, v in candidates:
        if v is None:
            continue
        val = value(v, exception=False)
        print(f"{name:40s} {val!s:>12} {str(v.lb):>10} {str(v.ub):>10}")

def main():
    nacl_feed = 70  # g/L
    
    print(f"Building model for NaCl feed = {nacl_feed} g/L...")
    m = build_and_initialize(nacl_feed * pyunits.g / pyunits.L)
    
    last_good, first_fail = find_recovery_limit(m, r_start=0.6, r_stop=0.5)

    print("\n--- Running Diagnostics on Failed State ---")
    dt = DiagnosticsToolbox(m)
    dt.report_numerical_issues()
    dt.display_variables_at_or_outside_bounds()
    dt.display_constraints_with_large_residuals()

    print_suspect_bounds(m)

if __name__ == "__main__":
    main()