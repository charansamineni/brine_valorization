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
    m.fs.product_objective = Objective(
        expr=m.fs.costing.LCOP
        + sum(
            (1 - m.fs.bpmed.product_mass_concentration[p])
            for p in m.fs.bpmed.product_mass_concentration
        )
    )

    # For debugging/unit inconsistency
    # dt = DiagnosticsToolbox(m)
    # dt.report_structural_issues()
    # dt.display_components_with_inconsistent_units()

    iscale.calculate_scaling_factors(m)
    assert degrees_of_freedom(m) == 0

    m.fs.feed.initialize()
    m.fs.dilute_feed.initialize()
    m.fs.bpmed.initialize()
    m.fs.costing.initialize()
    m.fs.bpmed.report()
    assert degrees_of_freedom(m) == 0


    m.fs.bpmed.set_optimization_operation()
    m.fs.bpmed.bpmed[0].electrical_stage_num.unfix()        # unfixing electrical stage for more flexibility- but it doesn't change from 1 for the different cases
    last_good, first_fail = find_recovery_limit(m)
    return m

def find_recovery_limit(m, r_start=0.74, r_end=0.75, r_step=0.001, scan_max_iter=3000):
    solver = get_solver()
    solver.options['max_iter'] = scan_max_iter
    solver.options['tol'] = 1e-6

    last_good_r = None
    for r in np.arange(r_start, r_end + r_step / 2, r_step):
        r = round(r, 4)
        m.fs.bpmed.nacl_recovery.fix(r)
        print(f"\n--- Trying nacl_recovery = {r} ---")

        try:
            result = solver.solve(m, tee=False)
            if result.solver.termination_condition != TerminationCondition.optimal:
                print(f"  -> NOT optimal at r={r}: {result.solver.termination_condition}")
                break
            print(f"  -> converged, LCOP={value(m.fs.costing.LCOP):.4f}, "
                  f"stages={value(m.fs.bpmed.bpmed[0].electrical_stage_num):.3f}")
            last_good_r = r
        except Exception as e:
            print(f"  -> EXCEPTION at r={r}: {e}")
            break

    print(f"\nLast successful recovery: {last_good_r}")
    print(f"First failing recovery: {r}")
    return last_good_r, r


def main():
    nacl_feed_vals = [150, 70]  # g/L
    nacl_recovery_vals = np.linspace(50, 75, 6)
#    nacl_recovery_vals = [0.7] # fractions
    solver = get_solver()
    solver.options['max_iter'] = 12000
    solver.options['tol'] = 1e-6

    results = []
    for nacl_feed in nacl_feed_vals:
        print(f"\n=== Building and initializing for NaCl feed = {nacl_feed} g/L ===")
        m = build_and_initialize(nacl_feed * pyunits.g / pyunits.L)

        for r in nacl_recovery_vals:
            m.fs.bpmed.nacl_recovery.fix(r)

            print(f"NaCl feed = {nacl_feed} g/L, nacl_recovery = {r}%, DOF = {degrees_of_freedom(m)}")
            result = solver.solve(m, tee=True)

            m.fs.dilute_feed.report()
            m.fs.bpmed.report()
            assert_optimal_termination(result)

            results.append(
                {
                    "NaCl_feed_gL": nacl_feed,
                    "nacl_recovery_pct": r,
                    "LCOP": value(m.fs.costing.LCOP),
                    "SEC": value(m.fs.costing.specific_energy_consumption),
                }
            )

    print("\n=== Summary ===")
    for row in results:
        print(row)

 
if __name__ == "__main__":

    main()




