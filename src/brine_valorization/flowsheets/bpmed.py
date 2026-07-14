from brine_valorization.costing.valorization_costing_block import (
    ValorizationCostingBlock,
)
from brine_valorization.unit_models.bpmed import (
    BPMED,
)

from watertap.core.solvers import get_solver
from pyomo.environ import (
    TransformationFactory,
    assert_optimal_termination,
)

import idaes.core.util.scaling as iscale
from reaktoro_enabled_watertap.utils.report_util import (
    build_report_table,
)
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
import sys

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

from watertap.property_models.water_prop_pack import WaterParameterBlock
from reaktoro_enabled_watertap.utils import scale_utils as scu
import numpy as np


def setup_feeds(NaCl=150 * pyunits.g / pyunits.L):
    mols_nacl = NaCl / (22.98977 + 35.45)
    mass_conc_na = mols_nacl * 22.98977
    mass_conc_cl = mols_nacl * 35.45
    mols_nacl_dilute = 0.01 / (22.98977 + 35.45)
    mass_conc_na_dilute = mols_nacl_dilute * 22.98977
    mass_conc_cl_dilute = mols_nacl_dilute * 35.45
    mcas_props = {
        "solute_list": ["Na_+", "Cl_-"],
        "mw_data": {
            "H2O": 18e-3,
            "Na_+": 22.98977e-3,
            "Cl_-": 35.45e-3,
        },
        "elec_mobility_data": {
            ("Liq", "Na_+"): 5.19e-8,
            ("Liq", "Cl_-"): 7.92e-8,
        },
        "charge": {"Na_+": 1, "Cl_-": -1},
        "diffusivity_data": {
            ("Liq", "Na_+"): 1.33e-9,
            ("Liq", "Cl_-"): 2.03e-9,
        },
    }
    feed_specs = {
        "temperature": 298.15,
        "pressure": 101325,
        "volumetric_flowrate": 1 * pyunits.L / pyunits.s,
        "ion_concentrations": {
            "Na_+": mass_conc_na,
            "Cl_-": mass_conc_cl,
        },  # "H_+": 1e-7, "OH_-": 1e-7},
    }
    feed_specs_dilute = {
        "temperature": 298.15,
        "pressure": 101325,
        "volumetric_flowrate": 1 * pyunits.L / pyunits.s,
        "ion_concentrations": {
            "Na_+": mass_conc_na_dilute,
            "Cl_-": mass_conc_cl_dilute,
        },  # "H_+": 1e-7, "OH_-": 1e-7},
    }
    m = ConcreteModel()
    m.fs = FlowsheetBlock()

    mcas_props["activity_coefficient_model"] = ActivityCoefficientModel.ideal
    mcas_props["density_calculation"] = DensityCalculation.constant
    m.fs.properties = MCASWEParameterBlock(**mcas_props)
    m.fs.dilute_properties = MCASWEParameterBlock(**mcas_props)
    m.fs.feed = MultiCompFeed(
        default_property_package=m.fs.properties,
        reconcile_using_reaktoro=False,
        **feed_specs,
    )
    m.fs.nacl_concentration = Var(initialize=value(NaCl), units=pyunits.g / pyunits.L)
    iscale.set_scaling_factor(m.fs.nacl_concentration, 1)

    @m.fs.Constraint(["Na_+", "Cl_-"])
    def ion_conc_constraint(b, ion):
        if ion == "Na_+":

            return b.nacl_concentration * 22.98977 / (
                22.98977 + 35.45
            ) == pyunits.convert(
                b.feed.feed.properties[0].flow_mass_phase_comp["Liq", ion]
                / b.feed.feed.properties[0].flow_vol_phase["Liq"],
                to_units=pyunits.g / pyunits.L,
            )
        elif ion == "Cl_-":
            return b.nacl_concentration * 35.45 / (22.98977 + 35.45) == pyunits.convert(
                b.feed.feed.properties[0].flow_mass_phase_comp["Liq", ion]
                / b.feed.feed.properties[0].flow_vol_phase["Liq"],
                to_units=pyunits.g / pyunits.L,
            )

    iscale.set_scaling_factor(m.fs.ion_conc_constraint["Na_+"], 1)
    iscale.set_scaling_factor(m.fs.ion_conc_constraint["Cl_-"], 1)

    m.fs.dilute_feed = MultiCompFeed(
        default_property_package=m.fs.dilute_properties,
        reconcile_using_reaktoro=False,
        **feed_specs_dilute,
    )
    return m


def get_constraint_vars(con):
    """Recursively get variables from a constraint expression and return lists of variables on the left and right hand sides of the constraint"""

    def get_vars_from_expr(var_list, expr):
        if isinstance(expr, float) == False and expr.is_expression_type():
            for arg in expr.args:
                get_arg = get_vars_from_expr(var_list, arg)
                if get_arg is not None:
                    var_list.append(get_arg)
        else:
            if expr.is_variable_type():
                var_list.append(expr)
            else:
                raise TypeError("Expression is not a variable or expression type")

    left, right = con.body.args
    left_vars, right_vars = [], []
    get_vars_from_expr(left_vars, left)
    get_vars_from_expr(right_vars, right)
    return left_vars, right_vars


def build_bpmed(
    NaCl=40, #g/L
    stages=1,
    add_feed_bleed_for_acid_base=True,
    add_feed_bleed_for_brine=True,
    add_mvc=False,
):
    m = setup_feeds(NaCl)
    if add_mvc:
        m.fs.water_properties_vapor = WaterParameterBlock()
        mvc_props = m.fs.water_properties_vapor
    else:
        mvc_props = None
    m.add_feed_bleed_for_acid_base = add_feed_bleed_for_acid_base
    m.add_feed_bleed_for_brine = add_feed_bleed_for_brine
    m.add_mvc = add_mvc
    m.fs.costing = ValorizationCostingBlock()
    m.fs.bpmed = BPMED(
        default_property_package=m.fs.properties,
        default_costing_package=m.fs.costing,
        mvc_vapor_prop_pack=mvc_props,
        add_mvc_concentrators=add_mvc,
        add_feed_bleed_for_acid_base=add_feed_bleed_for_acid_base,
        add_feed_bleed_for_brine=add_feed_bleed_for_brine,
        bpmed_stages=stages,
    )

    m.fs.feed.outlet.connect_to(m.fs.bpmed.brine_inlet)
    m.fs.dilute_feed.outlet.connect_to(m.fs.bpmed.low_tds_water_inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

    m.fs.feed.fix_and_scale()
    m.fs.dilute_feed.fix_and_scale()
    m.fs.bpmed.fix_and_scale()
    m.fs.costing.cost_process()
    # m.fs.costing.add_annual_product_generation(
    #     sum(m.fs.bpmed.flow_mass_product[p] for p in m.fs.bpmed.flow_mass_product)
    # )
    if add_mvc == True:
        process = "mvc"
    else:
        process = "bpmed"
    for p, product in m.fs.bpmed.flow_mass_product:
        if p == process:
            m.fs.costing.add_LCOP(
                m.fs.bpmed.flow_mass_product[p, product], "LCOP_" + product
            )
            m.fs.costing.add_mass_based_specific_energy_consumption(
                m.fs.bpmed.flow_mass_product[p, product], "SEC_" + product
            )

    total_product = [-m.fs.dilute_feed.feed.properties[0].flow_vol_phase["Liq"]]
    if m.add_mvc:
        total_product.append(
            m.fs.bpmed.distillate_mixer.mixed_state[0].flow_vol_phase["Liq"]
        )

    m.fs.total_product_water = Var(initialize=1, units=pyunits.m**3 / pyunits.s)
    m.fs.eq_total_product_flow = Constraint(
        expr=m.fs.total_product_water == sum(total_product)
    )

    iscale.set_scaling_factor(m.fs.total_product_water, 1e-3)
    iscale.constraint_scaling_transform(m.fs.eq_total_product_flow, 1e-3)

    m.fs.costing.add_LCOW(
        flow_rate=m.fs.total_product_water,
    )

    if m.add_mvc:
        m.fs.objective = Objective(
            expr=m.fs.costing.LCOP_HCl + m.fs.costing.LCOP_NaOH  # + m.fs.costing.LCOW
        )
    else:
        m.fs.product_objective = Objective(
            expr=m.fs.costing.LCOP_HCl
            + m.fs.costing.LCOP_NaOH
            + (
                sum(
                    (1 - m.fs.bpmed.product_mass_concentration[p])
                    for p in m.fs.bpmed.flow_mass_product
                )
            )
            ** 2
        )

    iscale.calculate_scaling_factors(m)

    scu.scale_costing_block(m.fs.costing)
    return m


def add_expected_quality_constraint(m):
    m.fs.naoh_min_constraint = Constraint(
        expr=m.fs.bpmed.product_mass_concentration["bpmed", "NaOH"]
        == m.fs.feed.feed.properties[0].mass_frac_phase_comp["Liq", "Na_+"] / 2.2
    )
    m.fs.hcl_min_constraint = Constraint(
        expr=m.fs.bpmed.product_mass_concentration["bpmed", "HCl"]
        == m.fs.feed.feed.properties[0].mass_frac_phase_comp["Liq", "Cl_-"] / 2.2
    )
    # m.fs.naoh_max_constraint = Constraint(
    #     expr=m.fs.bpmed.product_mass_concentration["bpmed", "NaOH"]
    #     <= m.fs.feed.feed.properties[0].mass_frac_phase_comp["Liq", "Na_+"] / 1.8
    # )
    # m.fs.hcl_max_constraint = Constraint(
    #     expr=m.fs.bpmed.product_mass_concentration["bpmed", "HCl"]
    #     <= m.fs.feed.feed.properties[0].mass_frac_phase_comp["Liq", "Cl_-"] / 1.8
    # )
    iscale.constraint_scaling_transform(m.fs.naoh_min_constraint, 1e-2)
    iscale.constraint_scaling_transform(m.fs.hcl_min_constraint, 1e-2)

    # iscale.constraint_scaling_transform(m.fs.naoh_max_constraint, 1e-2)
    # iscale.constraint_scaling_transform(m.fs.hcl_max_constraint, 1e-2)


def add_equal_quality_constraints(m):
    m.fs.equal_quality_constraint = Constraint(
        expr=m.fs.bpmed.product_mass_concentration["bpmed", "NaOH"]
        == m.fs.bpmed.product_mass_concentration["bpmed", "HCl"]
    )
    iscale.set_scaling_factor(m.fs.equal_quality_constraint, 1)


def initialize(m, **kwargs):
    m.fs.ion_conc_constraint["Na_+"].deactivate()
    m.fs.ion_conc_constraint["Na_+"].deactivate()
    print('DOF before initialize: ', degrees_of_freedom(m))
    assert int(degrees_of_freedom(m)) == 0
    m.fs.feed.initialize()
    m.fs.dilute_feed.initialize()
    m.fs.bpmed.initialize()
    m.fs.costing.initialize()
    assert int(degrees_of_freedom(m)) == 0
    # switch to fixing the NaCl concentration, unfix the ion concentrations
    m.fs.nacl_concentration.fix()
    m.fs.feed.feed.properties[0].conc_mass_phase_comp["Liq", "Na_+"].unfix()
    m.fs.feed.feed.properties[0].conc_mass_phase_comp["Liq", "Cl_-"].unfix()
    m.fs.ion_conc_constraint["Na_+"].activate()
    m.fs.ion_conc_constraint["Cl_-"].activate()
    assert int(degrees_of_freedom(m)) == 0
    m.fs.bpmed.set_optimization_operation()
    m.fs.bpmed.nacl_recovery.unfix()
    if m.add_feed_bleed_for_acid_base and m.add_mvc == False:
        add_equal_quality_constraints(m)

    m.fs.bpmed.nacl_recovery.fix()

    if m.add_mvc == True:
        m.fs.bpmed.acidate_mvc.recovery.fix()
        m.fs.bpmed.basate_mvc.recovery.fix()
    solve_model(m)

    if m.add_mvc == True:
        m.fs.bpmed.activate_product_quality_constraints({"HCl": 0.3, "NaOH": 0.25})
        m.fs.bpmed.acidate_mvc.recovery.unfix()
        m.fs.bpmed.basate_mvc.recovery.unfix()
        m.fs.bpmed.acidate_mvc.recovery.setlb(0.25)
        m.fs.bpmed.basate_mvc.recovery.setlb(0.25)
    m.fs.bpmed.nacl_recovery.unfix()
    m.fs.bpmed.nacl_recovery.setub(0.75)

    solve_model(m)
    # check_jac(m)
    # dg = DiagnosticsToolbox(m)
    # dg.report_numerical_issues()

    # dg.report_structural_issues()
    # dg.display_constraints_with_large_residuals()
    # dg.display_variables_at_or_outside_bounds()
    # dg.display_constraints_with_extreme_jacobians()
    # dg.display_variables_with_extreme_jacobians()
    # dg.display_constraints_with_canceling_terms()
    # dg.display_constraints_with_mismatched_terms()

    m.fs.bpmed.nacl_recovery.unfix()
    m.fs.dilute_feed.feed.properties[0].flow_vol_phase["Liq"].unfix()
    curflow = m.fs.dilute_feed.feed.properties[0].flow_vol_phase["Liq"]
    m.fs.dilute_feed.feed.properties[0].flow_vol_phase["Liq"].setlb(
        value(curflow * 0.01)
    )
    m.fs.dilute_feed.feed.properties[0].flow_vol_phase["Liq"].setub(value(curflow * 1))
    solve_model(m)


def check_jac(m, print_extreme_jacobian_values=True):
    jac, jac_scaled, nlp = iscale.constraint_autoscale_large_jac(m, min_scale=1e-8)
    try:
        cond_number = iscale.jacobian_cond(m, jac=jac_scaled) / 1e10
        print("--------------------------")
        print("COND NUMBER:", cond_number)
    except:
        print("Cond number failed")
        cond_number = None
    if print_extreme_jacobian_values:
        print("--------------------------")
        print("Extreme Jacobian entries:")
        extreme_entries = iscale.extreme_jacobian_entries(
            m, jac=jac_scaled, nlp=nlp, zero=1e-20, large=100
        )
        for val, var, con in extreme_entries:
            print(val, var.name, con.name)
        print("--------------------------")
        print("Extreme Jacobian columns:")
        extreme_cols = iscale.extreme_jacobian_columns(
            m, jac=jac_scaled, nlp=nlp, small=1e-3
        )
        for val, var in extreme_cols:
            print(val, var.name)
        print("------------------------")
        print("Extreme Jacobian rows:")
        extreme_rows = iscale.extreme_jacobian_rows(
            m, jac=jac_scaled, nlp=nlp, small=1e-3
        )
        for val, con in extreme_rows:
            print(val, con.name)

    for i in iscale.list_unscaled_variables(m):
        print("Var with no scale:", i)
    for i in iscale.list_unscaled_constraints(m):
        print("Constraint with no scale:", i)
    for var, scale in iscale.badly_scaled_var_generator(m):
        print(
            "Badly scaled variable:",
            var.name,
            var.value,
            iscale.get_scaling_factor(var),
        )
    return cond_number


def report(m):
    m.fs.feed.report()
    m.fs.dilute_feed.report()
    m.fs.bpmed.report()
    report_global_state(m)


def solve_model(m, **kwargs):
    # Replaced get_cyipopt_watertap_solver(linear_solver="ma27", max_iter=5000) from reaktoro_pse.
    # If convergence is poor, consider switching back to the cyipopt solver with MA27.
    solver = get_solver(options={"max_iter": 5000})
    result = solver.solve(m, tee=True)
    report(m)

    # assert_optimal_termination(result)
    return result


def report_global_state(m):
    data_dict = {"Global results": {}}
    data_dict["Global results"]["DOfs"] = int(degrees_of_freedom(m))
    data_dict["Global results"]["LCOP_HCl"] = m.fs.costing.LCOP_HCl
    data_dict["Global results"]["LCOP_NaOH"] = m.fs.costing.LCOP_NaOH
    data_dict["Global results"]["NaCl feed"] = m.fs.nacl_concentration
    data_dict["Global results"]["LCOW"] = m.fs.costing.LCOW
    data_dict["Global results"]["Product water flow"] = m.fs.total_product_water
    build_report_table("Global results", data_dict)


def show_fixed_vars(m):
    for v in m.component_data_objects(Var, active=True):
        if v.fixed:
            print(v.name, value(v))


if __name__ == "__main__":
    m = build_bpmed(
        NaCl=360,
        stages=1,
        add_feed_bleed_for_acid_base=True,
        add_feed_bleed_for_brine=False,
        add_mvc=False,
    )
    initialize(m)
    # for target in [
    #     0.011,
    #     0.012,
    #     0.013,
    #     0.014,
    #     0.015,
    #     0.016,
    #     0.017,
    #     0.018,
    #     0.019,
    #     0.02,
    # ]:
    #     print(f"Solving for target: {target}")
    #     m.fs.bpmed.activate_product_quality_constraints(target)
    #     solve_model(m)

    for nacl in np.linspace(360, 20, 18):
        print(f"Solving for NaCl concentration: {nacl} g/L")
        m.fs.nacl_concentration.fix(nacl)
        #     # m.fs.feed.feed.properties[0].conc_mass_phase_comp["Liq", "Na_+"].fix(
        #     #     nacl * 22.98977 / (22.98977 + 35.45)
        #     # )
        #     # m.fs.feed.feed.properties[0].conc_mass_phase_comp["Liq", "Cl_-"].fix(
        #     #     nacl * 35.45 / (22.98977 + 35.45)
        #     # )
        solve_model(m)
    # #     # check_jac(m)
