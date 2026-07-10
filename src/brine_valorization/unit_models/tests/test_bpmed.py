from brackish_valorization_reaktoro.property_models.tests.test_mcas_with_enthalpy import (
    build_case,
)
from brackish_valorization_reaktoro.costing.valorization_costing_block import (
    ValorizationCostingBlock,
)
import pytest
from brackish_valorization_reaktoro.unit_models.bpmed import (
    BPMED,
)

from reaktoro_pse.core.util_classes.cyipopt_solver import (
    get_cyipopt_watertap_solver,
)
from pyomo.environ import (
    TransformationFactory,
    assert_optimal_termination,
)

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
from brackish_valorization_reaktoro.property_models.mcas_with_enthalpy import (
    MCASWEParameterBlock,
)
from watertap.property_models.multicomp_aq_sol_prop_pack import (
    ActivityCoefficientModel,
    DensityCalculation,
)
from reaktoro_enabled_watertap.water_sources.source_water_importer import (
    get_source_water_data,
)

from watertap.costing import WaterTAPCosting
from idaes.core.util.model_statistics import degrees_of_freedom
import sys

try:
    # needed for windows dealign with ohm symbol encoding
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
except:
    pass

from watertap.property_models.water_prop_pack import WaterParameterBlock
from idaes.core.util.model_diagnostics import DiagnosticsToolbox

from watertap.core.solvers import get_solver

uthor__ = "Alexander V. Dudchenko"


def build_BPMED_feed_cases(water, reconcile_using_reaktoro=False):
    mcas_props, feed_specs = get_source_water_data(f"{water}.yaml")
    m = ConcreteModel()
    m.fs = FlowsheetBlock()

    mcas_props["activity_coefficient_model"] = ActivityCoefficientModel.ideal
    mcas_props["density_calculation"] = DensityCalculation.constant
    feed_specs["ion_concentrations"]["Na_+"] = (
        feed_specs["ion_concentrations"]["Na_+"] * 50
    )

    feed_specs["ion_concentrations"]["Cl_-"] = (
        feed_specs["ion_concentrations"]["Cl_-"] * 50
    )
    m.fs.properties = MCASWEParameterBlock(**mcas_props)
    m.fs.dilute_properties = MCASWEParameterBlock(**mcas_props)
    m.fs.feed = MultiCompFeed(
        default_property_package=m.fs.properties,
        reconcile_using_reaktoro=reconcile_using_reaktoro,
        **feed_specs,
    )
    feed_specs["ion_concentrations"] = {
        ion: value / 100 for ion, value in feed_specs["ion_concentrations"].items()
    }
    m.fs.dilute_feed = MultiCompFeed(
        default_property_package=m.fs.dilute_properties,
        reconcile_using_reaktoro=reconcile_using_reaktoro,
        **feed_specs,
    )
    m.fs.feed.fix_and_scale()
    m.fs.feed.report()
    return m


@pytest.mark.core
@pytest.mark.component
def test_BPMED_direct_pass_through():
    m = build_BPMED_feed_cases("USDA_brackish", True)

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
    iscale.calculate_scaling_factors(m)
    assert degrees_of_freedom(m) == 0

    m.fs.feed.initialize()
    m.fs.dilute_feed.initialize()
    m.fs.bpmed.initialize()
    m.fs.costing.initialize()
    m.fs.bpmed.report()
    assert degrees_of_freedom(m) == 0
    m.fs.dilute_feed.feed.properties[0].flow_vol_phase["Liq"].unfix()
    m.fs.bpmed.set_optimization_operation()
    for r in [0.6]:
        m.fs.bpmed.nacl_recovery.fix(r)
        # m.fs.bpmed.activate_product_quality_constraints(target_concentration=0.01)
        print(degrees_of_freedom(m))
        solver = get_solver()
        result = solver.solve(m, tee=True)
        m.fs.dilute_feed.report()
        m.fs.bpmed.report()
        assert_optimal_termination(result)


@pytest.mark.core
@pytest.mark.component
def test_BPMED_direct_pass__multi_stage():
    m = build_BPMED_feed_cases("USDA_brackish", True)

    m.fs.costing = ValorizationCostingBlock()
    m.fs.bpmed = BPMED(
        default_property_package=m.fs.properties,
        default_costing_package=m.fs.costing,
        bpmed_stages=3,
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
    iscale.calculate_scaling_factors(m)
    # dg.display_potential_evaluation_errors()
    # dg.display_variables_fixed_to_zero()
    print(degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0

    m.fs.feed.initialize()
    m.fs.dilute_feed.initialize()
    m.fs.bpmed.initialize()
    m.fs.costing.initialize()
    m.fs.bpmed.report()
    print(degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0


@pytest.mark.core
@pytest.mark.component
def test_BPMED_feed_bleed():
    m = build_BPMED_feed_cases("USDA_brackish", True)

    m.fs.costing = ValorizationCostingBlock()
    m.fs.bpmed = BPMED(
        default_property_package=m.fs.properties,
        add_feed_bleed_for_acid_base=True,
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
    iscale.calculate_scaling_factors(m)
    print(degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0

    m.fs.feed.initialize()
    m.fs.dilute_feed.initialize()
    m.fs.bpmed.initialize()
    m.fs.bpmed.report()
    print(degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0


@pytest.mark.core
@pytest.mark.component
def test_BPMED_feed_mvc():
    m = build_BPMED_feed_cases("USDA_brackish", True)

    m.fs.costing = ValorizationCostingBlock()
    m.fs.water_properties_vapor = WaterParameterBlock()
    m.fs.bpmed = BPMED(
        default_property_package=m.fs.properties,
        add_mvc_concentrators=True,
        mvc_vapor_prop_pack=m.fs.water_properties_vapor,
        default_costing_package=m.fs.costing,
    )

    m.fs.feed.outlet.connect_to(m.fs.bpmed.brine_inlet)
    m.fs.dilute_feed.outlet.connect_to(m.fs.bpmed.low_tds_water_inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

    m.fs.feed.fix_and_scale()
    m.fs.dilute_feed.fix_and_scale()
    m.fs.bpmed.fix_and_scale()
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
    # dg = DiagnosticsToolbox(m)
    # dg.display_underconstrained_set()
    # dg.display_overconstrained_set()
    # print(degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0

    m.fs.feed.initialize()
    m.fs.dilute_feed.initialize()
    m.fs.bpmed.initialize()
    m.fs.bpmed.report()
    print(degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0
