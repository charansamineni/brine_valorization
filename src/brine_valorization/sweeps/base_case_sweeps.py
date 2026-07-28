from brine_valorization.costing.valorization_costing_block import (
    ValorizationCostingBlock,
)
import pytest
from brine_valorization.unit_models.bpmed import (
    BPMED,
)

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

    # m.fs.recovery_objective = Objective(
    #     expr = - (m.fs.bpmed.nacl_recovery)
    # )
    # expr = - (nacl_recovery) to maximize recovery



    iscale.calculate_scaling_factors(m)
    assert degrees_of_freedom(m) == 0

    m.fs.feed.initialize()
    m.fs.dilute_feed.initialize()


    try:
        m.fs.bpmed.initialize()
    except InitializationError as e:
        print(f"\n*** m.fs.bpmed.initialize() failed: {e} ***\n")
        from idaes.core.util.model_diagnostics import DiagnosticsToolbox
        dt = DiagnosticsToolbox(m)
        dt.report_numerical_issues()
        dt.display_variables_at_or_outside_bounds()
        dt.display_constraints_with_large_residuals()
        raise

#    m.fs.bpmed.initialize()
    m.fs.bpmed.report()
    assert degrees_of_freedom(m) == 0

    solver = get_solver()

    print("Solving fixed-operation model...")
    result = solver.solve(m, tee=True)
    print(result.solver.termination_condition)

    m.fs.bpmed.set_optimization_operation()
 #   m.fs.bpmed.bpmed[0].electrical_stage_num.unfix() 
      # idaes model statistics --> can check electrical_stage_num to see if it's fixed or unfixed
    # m.fs.bpmed.bpmed[0].cell_length.fix(2.0)
    # m.fs.bpmed.bpmed[0].cell_width.fix(0.5)
    
    m.fs.bpmed.bpmed[0].cell_triplet_num.unfix()
    
    # Give it generous bounds so it can build a massive stack if needed for high recovery
    m.fs.bpmed.bpmed[0].cell_triplet_num.setlb(50)
    m.fs.bpmed.bpmed[0].cell_triplet_num.setub(10000)
    return m

def find_recovery_limit(m, r_start=0.5, r_stop=0.85, coarse_step=0.02, fine_step=0.002,
                         scan_max_iter=7000):
    solver = get_solver()
    solver.options['max_iter'] = scan_max_iter
    solver.options['tol'] = 1e-6

    last_good_r = r_start
    r = r_start

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

    # Coarse ramp with warm-starting from the last good point
    r = r_start
    while r < r_stop:
        r_next = round(r + coarse_step, 4)
        if try_solve(r_next):
            last_good_r = r_next
            r = r_next
        else:
            break
    else:
        print("Reached r_stop without failure.")
        return last_good_r, None

    # Fine bisection between last_good_r and r_next (the first coarse failure)
    first_fail_r = r_next
    lo, hi = last_good_r, first_fail_r
    while hi - lo > fine_step:
        mid = round((lo + hi) / 2, 4)
        if try_solve(mid):
            last_good_r = lo = mid
        else:
            first_fail_r = hi = mid

    print(f"\nLast successful recovery: {last_good_r}")
    print(f"First failing recovery:   {first_fail_r}")

    # Leave the model fixed at the failing point for diagnostics
    m.fs.bpmed.nacl_recovery.fix(first_fail_r)
    solver.solve(m, tee=False)  # re-solve so m holds the failed iterate
    return last_good_r, first_fail_r


def print_suspect_bounds(m):
    b = m.fs.bpmed
    candidates = [
        ("low_tds_splitter.split_fraction[acidate]", b.low_tds_splitter.split_fraction[0, "acidate"]),
        ("acidate_splitter.split_fraction[recycle]", getattr(b, "acidate_splitter", None) and b.acidate_splitter.split_fraction[0, "recycle"]),
        ("basate_splitter.split_fraction[recycle]", getattr(b, "basate_splitter", None) and b.basate_splitter.split_fraction[0, "recycle"]),
        ("voltage_applied", b.bpmed[0].voltage_applied[0]),
        ("cell_length", b.bpmed[0].cell_length),
        ("cell_triplet_num", b.bpmed[0].cell_triplet_num),
        ("velocity_diluate", b.bpmed[0].velocity_diluate[0, 0]),
        ("velocity_acidate", b.bpmed[0].velocity_acidate[0, 0]),
        ("velocity_basate", b.bpmed[0].velocity_basate[0, 0]),
        ("concentration aem", b.bpmed[0].salt_conc_aem_x[0, 1]),
        ("concentration cem", b.bpmed[0].salt_conc_cem_x[0, 1]),
        ("concentration dilu", b.bpmed[0].salt_conc_dilu_x[0, 1])
    ]
    print(f"\n{'Variable':40s} {'Value':>12s} {'LB':>10s} {'UB':>10s}")
    for name, v in candidates:
        if v is None:
            continue
        val = value(v, exception=False)
        print(f"{name:40s} {val!s:>12} {str(v.lb):>10} {str(v.ub):>10}")


def main():
 #   nacl_feed_vals = [150, 70]  # g/L
    nacl_feed = 70
#    nacl_recovery_vals = np.linspace(.70, .80, 6)
    nacl_recovery_val = 0.7 # fractions
    solver = get_solver()
    solver.options['max_iter'] = 5000
    solver.options['tol'] = 1e-6

    m = build_and_initialize(nacl_feed * pyunits.g / pyunits.L)

    m.fs.bpmed.nacl_recovery.fix(0.7)
    solver.solve(m, tee=False)

    m.fs.product_objective.deactivate()

    m.fs.bpmed.nacl_recovery.unfix()
    m.fs.recovery_objective = Objective(
    expr = - (m.fs.bpmed.nacl_recovery)
    )
    solver.solve(m, tee=False)

    m.fs.bpmed.nacl_recovery.display()

 #   m.fs.bpmed.nacl_recovery.fix(nacl_recovery_val)

    
 #   last_good, first_fail = find_recovery_limit(m, r_start=0.5, r_stop=0.85)

    dt = DiagnosticsToolbox(m)
    dt.report_numerical_issues()
    dt.display_variables_at_or_outside_bounds()
    dt.display_constraints_with_large_residuals()
    # check for constraints that are also at/close to their bounds ("body")

    print_suspect_bounds(m)


 
if __name__ == "__main__":

    main()




