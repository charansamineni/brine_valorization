from brine_valorization.costing.valorization_costing_block import (
    ValorizationCostingBlock,
)
import pytest
from brine_valorization.unit_models.bpmed import (
    BPMED,
)
from pyomo.opt import TerminationCondition
from functools import partial

from pyomo.environ import (
    TransformationFactory,
    assert_optimal_termination,
)
from datetime import datetime

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

from parameter_sweep import (LinearSample, parameter_sweep)


def build_BPMED_feed_cases(NaCl=70 * pyunits.g / pyunits.L):
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



def build_and_solve(feed=70, **kwargs):


    m = build_BPMED_feed_cases(NaCl=feed * pyunits.g / pyunits.L)

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

    # --- added: total product water flow ---
    m.fs.total_product_water = Var(initialize=1, units=pyunits.m**3 / pyunits.s)
    m.fs.eq_total_product_flow = Constraint(
        expr=m.fs.total_product_water
        == -m.fs.dilute_feed.feed.properties[0].flow_vol_phase["Liq"]
    )
    iscale.set_scaling_factor(m.fs.total_product_water, 1e-3)
    iscale.constraint_scaling_transform(m.fs.eq_total_product_flow, 1e-3)
    # ----------------------------------------

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

    # added objective to only maximize product concentration? (previously, getting 6-7% with NaOH conc higher, but it peaked at recovery of 0.675 recovery)
    # m.fs.product_conc_objective = Objective(
    #     expr = (- (m.fs.bpmed.product_mass_concentration[p]) for p in m.fs.bpmed.product_mass_concentration)
    # )


    iscale.calculate_scaling_factors(m)
    assert degrees_of_freedom(m) == 0

    m.fs.feed.initialize()
    m.fs.dilute_feed.initialize()
    m.fs.bpmed.initialize()
    m.fs.costing.initialize()

    m.fs.bpmed.report()
    assert degrees_of_freedom(m) == 0
    m.fs.bpmed.bpmed[0].electrical_stage_num.unfix()
    m.fs.bpmed.set_optimization_operation()

# solving once at 0.7
    solver =get_solver()
    m.fs.bpmed.nacl_recovery.fix(0.7)
#     res_warmup = solver.solve(m, tee=False)
# #   assert_optimal_termination(res_warmup)
#     if res_warmup.solver.termination_condition != TerminationCondition.optimal:
#         print(f"Warning: Initial warm-up solve at feed={feed} resulted in {res_warmup.solver.termination_condition}")

#    m.fs.bpmed.nacl_recovery.unfix()

    solver.options['max_iter'] = 7000
    solver.options['tol'] = 1e-6
    result = solver.solve(m, tee=True)

    print("Unit capital cost:",
      value(m.fs.bpmed.bpmed[0].costing.capital_cost))

    print("Aggregate capital cost:",
        value(m.fs.costing.aggregate_capital_cost))

    print("Total capital cost:",
        value(m.fs.costing.total_capital_cost))

    print("LCOP:",
        value(m.fs.costing.LCOP))

    return m

def build_sweep_params(m, num_samples=36):

    sweep_params = {}
    sweep_params["NaCl Recovery"] = LinearSample(
        m.fs.bpmed.nacl_recovery, 0.5, 0.85, num_samples

    )
    return sweep_params

def add_output(outputs, name, obj):
    """
    Add a scalar or indexed Pyomo Var/Expression to the outputs dictionary.
    """

    if obj.is_indexed():
        for idx in obj:
            if isinstance(idx, tuple):
                idx_str = "_".join(map(str, idx))
            else:
                idx_str = str(idx)

            outputs[f"{name}[{idx_str}]"] = obj[idx]
    else:
        outputs[name] = obj

def build_outputs(m):


    outputs = {}

    # ----------------------------
    # Performance
    # ----------------------------
    add_output(outputs, "LCOP", m.fs.costing.LCOP)
    add_output(outputs, "Specific Energy Consumption",
               m.fs.costing.specific_energy_consumption)
    add_output(outputs, "Annual Product Generation",
               m.fs.costing.annual_product_generation)
    add_output(outputs, "NaCl Recovery Check", m.fs.bpmed.nacl_recovery) # just in case lol

    # ----------------------------
    # Flows
    # ----------------------------
    add_output(outputs, "Total Product Water",
               m.fs.total_product_water)
# generally for flow_mass_product, add_output will run through indexed components 
    add_output(outputs, "Flow Mass Product", m.fs.bpmed.flow_mass_product)

    add_output(outputs, "Feed Inlet Flow Vol",
           m.fs.feed.feed.properties[0].flow_vol_phase["Liq"])

    # total_product_water is the negative of this
    add_output(outputs, "Dilute Feed Inlet Flow Vol",
           m.fs.dilute_feed.feed.properties[0].flow_vol_phase["Liq"])
    

    # ----------------------------
    # Feed Conditions
    # ----------------------------
    add_output(outputs, "Feed Mass Frac Na", m.fs.feed.feed.properties[0].mass_frac_phase_comp["Liq", "Na_+"])
    add_output(outputs, "Feed Mass Frac Cl", m.fs.feed.feed.properties[0].mass_frac_phase_comp["Liq", "Cl_-"])
    add_output(outputs, "Feed Temperature", m.fs.feed.feed.properties[0].temperature)

    

    # ----------------------------
    # Costing
    # ----------------------------
    add_output(outputs, "Total Capital Cost",
               m.fs.costing.total_capital_cost)
    add_output(outputs, "Aggregate Capital Cost",
               m.fs.costing.aggregate_capital_cost)
    add_output(outputs, "Aggregate Direct Capital Cost",
               m.fs.costing.aggregate_direct_capital_cost)
    add_output(outputs, "Aggregate Fixed Operating Cost",
               m.fs.costing.aggregate_fixed_operating_cost)
    add_output(outputs, "Aggregate Variable Operating Cost",
               m.fs.costing.aggregate_variable_operating_cost)
    add_output(outputs, "Aggregate Flow Electricity",
               m.fs.costing.aggregate_flow_electricity)
    add_output(outputs, "Aggregate Electricity Cost",
               m.fs.costing.aggregate_flow_costs["electricity"])
    add_output(outputs, "Total Operating Cost",
               m.fs.costing.total_operating_cost)
    add_output(outputs, "Total Fixed Operating Cost",
               m.fs.costing.total_fixed_operating_cost)
    add_output(outputs, "Total Variable Operating Cost",
               m.fs.costing.total_variable_operating_cost)
    add_output(outputs, "Total Annualized Cost",
               m.fs.costing.total_annualized_cost)
    add_output(outputs, "Maintenance Labor Chemical Operating Cost",
           m.fs.costing.maintenance_labor_chemical_operating_cost)

    # ----------------------------
    # Costing parameters
    # ----------------------------
    add_output(outputs, "Capital Recovery Factor",
               m.fs.costing.capital_recovery_factor)
    add_output(outputs, "Electricity Cost",
               m.fs.costing.electricity_cost)
    add_output(outputs, "Electrical Carbon Intensity",
               m.fs.costing.electrical_carbon_intensity)
    add_output(outputs, "Plant Lifetime",
               m.fs.costing.plant_lifetime)
    add_output(outputs, "WACC",
               m.fs.costing.wacc)
    add_output(outputs, "TPEC",
               m.fs.costing.TPEC)
    add_output(outputs, "TIC",
               m.fs.costing.TIC)
    add_output(outputs, "Total Investment Factor",
               m.fs.costing.total_investment_factor)
    add_output(outputs, "Utilization Factor",
               m.fs.costing.utilization_factor)
    add_output(outputs, "Maintenance Labor Chemical Factor",
               m.fs.costing.maintenance_labor_chemical_factor)

    # ----------------------------
    # BPMED costing
    # ----------------------------
    add_output(outputs, "Membrane Capital Cost",
               m.fs.costing.bipolar_electrodialysis_costing.membrane_capital_cost)
    add_output(outputs, "Membrane Replacement Factor",
               m.fs.costing.bipolar_electrodialysis_costing.factor_membrane_replacement)
    add_output(outputs, "Stack Electrode Capital Cost",
               m.fs.costing.bipolar_electrodialysis_costing.stack_electrode_capital_cost)
    add_output(outputs, "Stack Electrode Replacement Factor",
               m.fs.costing.bipolar_electrodialysis_costing.factor_stack_electrode_replacement)
    add_output(outputs, "Low Pressure Pump Unit Cost",
               m.fs.costing.low_pressure_pump.unit_cost)

    # ----------------------------
    # BPMED variables
    # ----------------------------
    add_output(outputs, "Voltage Applied",
               m.fs.bpmed.bpmed[0].voltage_applied)

    add_output(outputs, "Total Areal Resistance",
               m.fs.bpmed.bpmed[0].total_areal_resistance_x)

    # add_output(outputs, "Current Density Limit",
    #            m.fs.bpmed.current_dens_lim_x)
    # only with has_Nernst_diffusion_layer and that initializes to False

    add_output(outputs, "Current Density Limit BPM",
               m.fs.bpmed.bpmed[0].current_dens_lim_bpem)

    add_output(outputs, "Product Mass Concentration",
               m.fs.bpmed.product_mass_concentration)

    add_output(outputs, "Cell Length",
               m.fs.bpmed.bpmed[0].cell_length)

    add_output(outputs, "Cell Width",
               m.fs.bpmed.bpmed[0].cell_width)

    add_output(outputs, "Cell Triplet Number",
               m.fs.bpmed.bpmed[0].cell_triplet_num)

    add_output(outputs, "Electrical Stage Number",
               m.fs.bpmed.bpmed[0].electrical_stage_num)

    add_output(outputs, "Electrode Resistance",
               m.fs.bpmed.bpmed[0].electrodes_resistance)

    add_output(outputs, "Pressure Drop",
               m.fs.bpmed.bpmed[0].pressure_drop)

    add_output(outputs, "Pressure Drop Total",
               m.fs.bpmed.bpmed[0].pressure_drop_total)

    return outputs


def custom_optimize(model):
    solver = get_solver()
    solver.options["max_iter"] = 5000  # Set your max iterations
    solver.options["tol"] = 1e-5  # Set tolerance
    return solver.solve(model, tee=False)

if __name__ == "__main__":
    num_samples = 36

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

    sweep_solver_options = {
        "max_iter": 7000,
        "tol": 1e-6,
    }

    for feed in [70, 150]:
        results_array, results_dict = parameter_sweep(
            build_model=partial(build_and_solve, feed=feed),
            build_sweep_params=build_sweep_params,
            build_sweep_params_kwargs={"num_samples": num_samples},
            build_outputs=build_outputs,
            optimize_function=custom_optimize,
            csv_results_file_name=f"updated_costing_vs_recovery_{feed}gL_{timestamp}.csv",
        )
    # results_array, results_dict = parameter_sweep(
    #     build_model=partial(build_and_solve, feed=70),
    #     build_sweep_params=build_sweep_params,
    #     build_sweep_params_kwargs={"num_samples": num_samples},
    #     build_outputs=build_outputs,
    #     optimize_function=custom_optimize,
    #     csv_results_file_name=f"test_costing_vs_recovery_{70}gL_{timestamp}.csv",
    # )
