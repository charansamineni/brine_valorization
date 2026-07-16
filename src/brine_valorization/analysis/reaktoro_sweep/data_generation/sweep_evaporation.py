# Predicts precipitated salts as a function of composition and evaporation percent
# Follows reaktoro_enabled_watertap's src/reaktoro_enabled_watertap/flowsheets/property_comparator/watertap_prop_comparison.py

import pandas as pd
from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    assert_optimal_termination,
    units as pyunits,
)
from idaes.core import FlowsheetBlock
from idaes.models.unit_models import Feed
import idaes.core.util.scaling as iscale
from idaes.core.util.model_statistics import degrees_of_freedom

from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
    ActivityCoefficientModel,
    DensityCalculation,
)
from reaktoro_pse.reaktoro_block import ReaktoroBlock
from reaktoro_pse.core.util_classes.cyipopt_solver import get_cyipopt_watertap_solver

__author__ = "Carson Tucker"

MINERAL_MOLAR_MASS = {
    "Halite": 58.44,
    "Gypsum": 172.17,
    "Anhydrite": 136.14,
    "Calcite": 100.09,
    "Sylvite": 74.55,
    "Epsomite": 246.47,
    "Mirabilite": 322.20,
    "Carnallite": 277.85,
}

DEFAULT_ION_PROPERTIES = {
    "Na_+": {"mw_kg_per_mol": 22.99e-3, "charge": 1},
    "Cl_-": {"mw_kg_per_mol": 35.45e-3, "charge": -1},
    "Ca_2+": {"mw_kg_per_mol": 40.08e-3, "charge": 2},
    "Mg_2+": {"mw_kg_per_mol": 24.31e-3, "charge": 2},
    "K_+": {"mw_kg_per_mol": 39.10e-3, "charge": 1},
    "SO4_2-": {"mw_kg_per_mol": 96.06e-3, "charge": -2},
    "HCO3_-": {"mw_kg_per_mol": 61.02e-3, "charge": -1},
    "Li_+": {"mw_kg_per_mol": 6.938e-3, "charge": 1},
}


def build_mcas_config(ion_properties):
    solute_list = list(ion_properties.keys())
    return dict(
        solute_list=solute_list,
        mw_data={ion: p["mw_kg_per_mol"] for ion, p in ion_properties.items()},
        charge={ion: p["charge"] for ion, p in ion_properties.items()},
        diffusivity_data={("Liq", ion): 0.0 for ion in solute_list},
        stokes_radius_data={ion: 0.0 for ion in solute_list},
        activity_coefficient_model=ActivityCoefficientModel.ideal,
        density_calculation=DensityCalculation.constant,
    )


def build_model(
    feed_concentrations, # g/L
    ion_properties=None,
    water_mass_flow=1.0, # kg/s
    temperature=298.15,
    pressure=101325,
    pH=7.0,
):
    m = ConcreteModel()
    m.fs = FlowsheetBlock()

    # Define feed
    ion_properties = ion_properties or DEFAULT_ION_PROPERTIES
    mcas_props = build_mcas_config({k: ion_properties[k] for k in feed_concentrations})

    m.fs.properties = MCASParameterBlock(**mcas_props)
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.feed_pH = Var(initialize=pH, bounds=(0, 14), units=pyunits.dimensionless)

    # Fix feed state
    m.fs.feed_pH.fix()
    m.fs.feed.properties[0].temperature.fix(temperature)
    m.fs.feed.properties[0].pressure.fix(pressure)
    for ion, conc in feed_concentrations.items():
        # Feed defined in g/L, but reaktoro will require molar flow as input. Conversion constraints written via MCAS
        m.fs.feed.properties[0].conc_mass_phase_comp["Liq", ion].fix(conc * pyunits.g / pyunits.L)
        m.fs.feed.properties[0].flow_mol_phase_comp["Liq", ion].unfix()

    m.fs.water_mass_flow = Var(
        initialize=water_mass_flow, units=pyunits.kg / pyunits.s
    )
    m.fs.water_mass_flow.fix(water_mass_flow)

    # Define evaporation percent variable and constraint
    m.fs.evaporation_percent = Var(initialize=0, bounds=(0, 100))
    m.fs.evaporation_percent.fix(0)

    @m.fs.Constraint()
    def eq_evaporation(fs):
        return m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"] == fs.water_mass_flow * (
            1 - fs.evaporation_percent / 100
        )

    m.fs.feed.properties[0].flow_mol_phase_comp["Liq", "H2O"].unfix()

    assert degrees_of_freedom(m) == 0, (
        f"Expected 0 DOF after building the feed, got {degrees_of_freedom(m)}. "
    )

    # Solve with 0 evaporation and to get the molar flow rates
    solver = get_cyipopt_watertap_solver()
    result = solver.solve(m, tee=False)
    assert_optimal_termination(result)

    scale_model(m)

    # Now need to unfix concentrations for varying evaporation percent
    for ion in feed_concentrations:
        m.fs.feed.properties[0].conc_mass_phase_comp["Liq", ion].unfix()
        m.fs.feed.properties[0].flow_mol_phase_comp["Liq", ion].fix()
    return m


def scale_model(m):
    for idx in m.fs.feed.properties[0].flow_mol_phase_comp:
        val = m.fs.feed.properties[0].flow_mol_phase_comp[idx].value
        if val:
            m.fs.properties.set_default_scaling(
                "flow_mol_phase_comp", 1 / val, index=idx
            )
    iscale.calculate_scaling_factors(m)


def add_precipitation_block(
    m,
    minerals,
    charge_balance_ion="Cl_-",
    database_file="pitzer.dat",
    activity_model="ActivityModelPitzer",
):

    # Define properties to track
    output_keys = [("speciesAmount", mineral) for mineral in minerals]
    output_keys += [("charge", None), ("pH", None)]
    m.fs.reaktoro_properties = Var(output_keys, initialize=1)

    # Scale output property variables
    for mineral in minerals:
        iscale.set_scaling_factor(
            m.fs.reaktoro_properties[("speciesAmount", mineral)], 1e5
        )
    iscale.set_scaling_factor(m.fs.reaktoro_properties[("charge", None)], 1e8)

    m.fs.eq_reaktoro_properties = ReaktoroBlock(
        aqueous_phase={
            "composition": m.fs.feed.properties[0].flow_mol_phase_comp,
            "convert_to_rkt_species": True,
            "activity_model": activity_model,
        },
        database_file=database_file,
        system_state={
            "temperature": m.fs.feed.properties[0].temperature,
            "pressure": m.fs.feed.properties[0].pressure,
            "pH": m.fs.feed_pH,
        },
        mineral_phase={"phase_components": minerals},
        outputs=m.fs.reaktoro_properties,
        assert_charge_neutrality=False,
        build_speciation_block=False, #Alread built with the feed
    )
    m.fs.eq_reaktoro_properties.initialize()

    # Unfix charge-balance ion molar flow and fix charge to 0
    m.fs.feed.properties[0].flow_mol_phase_comp["Liq", charge_balance_ion].unfix()
    m.fs.reaktoro_properties[("charge", None)].fix(0)

    assert degrees_of_freedom(m) == 0, (
        f"Expected 0 DOF after adding the Reaktoro block, got {degrees_of_freedom(m)}."
    )

    solver = get_cyipopt_watertap_solver()
    result = solver.solve(m, tee=False)
    assert_optimal_termination(result)
    return m


def sweep_evaporation(m, minerals, evaporation_percents, output_csv=None):
    solver = get_cyipopt_watertap_solver()
    records = []
    for evap_pct in evaporation_percents:
        m.fs.evaporation_percent.fix(evap_pct)
        result = solver.solve(m, tee=False)
        assert_optimal_termination(result)

        row = {
            "evaporation_percent": evap_pct,
            "pH": m.fs.reaktoro_properties[("pH", None)].value,
            "charge": m.fs.reaktoro_properties[("charge", None)].value,
        }
        total_solids_mass_g = 0.0
        for mineral in minerals:
            amount_mol = max(
                m.fs.reaktoro_properties[("speciesAmount", mineral)].value, 0.0
            )
            molar_mass = MINERAL_MOLAR_MASS.get(mineral)
            mass_g = amount_mol * molar_mass if molar_mass else None
            row[f"{mineral}_mol"] = amount_mol
            if mass_g is not None:
                row[f"{mineral}_g"] = mass_g
                total_solids_mass_g += mass_g
        row["total_solids_g"] = total_solids_mass_g
        records.append(row)
        print(
            f"Evaporation {evap_pct:6.2f}%: total solids {total_solids_mass_g:.4f} g/s"
        )

    df = pd.DataFrame.from_records(records)
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        print(f"Saved sweep results to {output_csv}")
    return df


if __name__ == "__main__":
    conc_g_per_L = {
        "Cl_-": 192.0,
        "SO4_2-": 23.3,
        # "H3BO3": 4.4, boric acid
        "Na_+": 93.2,
        "Ca_2+": 0.4,
        "Mg_2+": 12.3,
        "K_+": 22.0,
        "Li_+": 1.96,
    }
    # Added Halite, Sylvite, Sulfates, Carnallite based on paper - can add others
    minerals = ["Halite", "Gypsum", "Calcite", "Sylvite", "Carnallite", "Anhydrite"]
    save_folder = "src/brine_valorization/analysis/reaktoro_sweep/output/"

    m = build_model(conc_g_per_L, temperature=298.15, pressure=101325, pH=7.0)
    m = add_precipitation_block(m, minerals, charge_balance_ion="Cl_-")

    evaporation_percents = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    df = sweep_evaporation(
        m, minerals, evaporation_percents, output_csv=f"{save_folder}evaporation_solids_sweep.csv"
    )