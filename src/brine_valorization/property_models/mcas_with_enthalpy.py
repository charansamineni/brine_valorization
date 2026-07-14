from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterData,
    MCASStateBlockData,
    _MCASStateBlock,
)

# Import Pyomo libraries
from pyomo.environ import (
    Constraint,
    Expression,
    Reals,
    NonNegativeReals,
    log,
    Var,
    Param,
    Set,
    Suffix,
    value,
    check_optimal_termination,
    units as pyunits,
    exp,
)

from idaes.core import (
    declare_process_block_class,
)

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    EnergyBalanceType,
)


@declare_process_block_class("MCASWEParameterBlock")
class MCASWEParameterData(MCASParameterData):
    def build(self):
        super().build()
        self._state_block_class = MCASStateBlocWE


@declare_process_block_class("MCASStateBlocWE", block_class=_MCASStateBlock)
class MCASStateBlocWEData(MCASStateBlockData):

    def _enth_mass_phase(self):
        params = self.params
        if not hasattr(params, "enth_mass_param_A1"):
            # specific enthalpy parameters, 10-120 C, 0-120 g/kg, 0-12 MPa
            # Table 9 in Nayar et al. (2016)
            enth_mass_units = pyunits.J / pyunits.kg
            P_inv_units = pyunits.MPa**-1
            t_inv_units = pyunits.K**-1
            params.enth_mass_param_A1 = Var(
                within=Reals,
                initialize=996.7767,
                units=enth_mass_units * P_inv_units,
                doc="Specific enthalpy parameter A1",
            )
            params.enth_mass_param_A2 = Var(
                within=Reals,
                initialize=-3.2406,
                units=enth_mass_units * P_inv_units * t_inv_units,
                doc="Specific enthalpy parameter A2",
            )
            params.enth_mass_param_A3 = Var(
                within=Reals,
                initialize=0.0127,
                units=enth_mass_units * P_inv_units * t_inv_units**2,
                doc="Specific enthalpy parameter A3",
            )
            params.enth_mass_param_A4 = Var(
                within=Reals,
                initialize=-4.7723e-5,
                units=enth_mass_units * P_inv_units * t_inv_units**3,
                doc="Specific enthalpy parameter A4",
            )
            params.enth_mass_param_A5 = Var(
                within=Reals,
                initialize=-1.1748,
                units=enth_mass_units * P_inv_units,
                doc="Specific enthalpy parameter A5",
            )
            params.enth_mass_param_A6 = Var(
                within=Reals,
                initialize=0.01169,
                units=enth_mass_units * P_inv_units * t_inv_units,
                doc="Specific enthalpy parameter A6",
            )
            params.enth_mass_param_A7 = Var(
                within=Reals,
                initialize=-2.6185e-5,
                units=enth_mass_units * P_inv_units * t_inv_units**2,
                doc="Specific enthalpy parameter A7",
            )
            params.enth_mass_param_A8 = Var(
                within=Reals,
                initialize=7.0661e-8,
                units=enth_mass_units * P_inv_units * t_inv_units**3,
                doc="Specific enthalpy parameter A8",
            )
            params.enth_mass_param_B1 = Var(
                within=Reals,
                initialize=-2.34825e4,
                units=enth_mass_units,
                doc="Specific enthalpy parameter B1",
            )
            params.enth_mass_param_B2 = Var(
                within=Reals,
                initialize=3.15183e5,
                units=enth_mass_units,
                doc="Specific enthalpy parameter B2",
            )
            params.enth_mass_param_B3 = Var(
                within=Reals,
                initialize=2.80269e6,
                units=enth_mass_units,
                doc="Specific enthalpy parameter B3",
            )
            params.enth_mass_param_B4 = Var(
                within=Reals,
                initialize=-1.44606e7,
                units=enth_mass_units,
                doc="Specific enthalpy parameter B4",
            )
            params.enth_mass_param_B5 = Var(
                within=Reals,
                initialize=7.82607e3,
                units=enth_mass_units * t_inv_units,
                doc="Specific enthalpy parameter B5",
            )
            params.enth_mass_param_B6 = Var(
                within=Reals,
                initialize=-4.41733,
                units=enth_mass_units * t_inv_units**2,
                doc="Specific enthalpy parameter B6",
            )
            params.enth_mass_param_B7 = Var(
                within=Reals,
                initialize=2.1394e-1,
                units=enth_mass_units * t_inv_units**3,
                doc="Specific enthalpy parameter B7",
            )
            params.enth_mass_param_B8 = Var(
                within=Reals,
                initialize=-1.99108e4,
                units=enth_mass_units * t_inv_units,
                doc="Specific enthalpy parameter B8",
            )
            params.enth_mass_param_B9 = Var(
                within=Reals,
                initialize=2.77846e4,
                units=enth_mass_units * t_inv_units,
                doc="Specific enthalpy parameter B9",
            )
            params.enth_mass_param_B10 = Var(
                within=Reals,
                initialize=9.72801,
                units=enth_mass_units * t_inv_units**2,
                doc="Specific enthalpy parameter B10",
            )
            params.enth_mass_param_C1 = Var(
                within=Reals,
                initialize=141.355,
                units=enth_mass_units,
                doc="Specific enthalpy parameter C1",
            )
            params.enth_mass_param_C2 = Var(
                within=Reals,
                initialize=4202.07,
                units=enth_mass_units * t_inv_units,
                doc="Specific enthalpy parameter C2",
            )
            params.enth_mass_param_C3 = Var(
                within=Reals,
                initialize=-0.535,
                units=enth_mass_units * t_inv_units**2,
                doc="Specific enthalpy parameter C3",
            )
            params.enth_mass_param_C4 = Var(
                within=Reals,
                initialize=0.004,
                units=enth_mass_units * t_inv_units**3,
                doc="Specific enthalpy parameter C4",
            )

            for v in params.component_objects(Var):
                v.fix()

        self.enth_mass_phase = Var(
            self.params.phase_list,
            initialize=1e6,
            bounds=(1, 1e9),
            units=pyunits.J * pyunits.kg**-1,
            doc="Specific enthalpy",
        )

        # Nayar et al. (2016), eq. 25 and 26, 10-120 C, 0-120 g/kg, 0-12 MPa
        def rule_enth_mass_phase(b, p):
            # temperature in degC, but pyunits in K
            t = b.temperature - 273.15 * pyunits.K
            mass_solids = []
            for j in b.params.component_list:
                if j not in b.params.solvent_set:
                    mass_solids.append(b.flow_mass_phase_comp[p, j])
            mass_solution = mass_solids[:]
            mass_solution.append(b.flow_mass_phase_comp[p, "H2O"])
            S_kg_kg = sum(mass_solids) / sum(mass_solution)
            S_g_kg = S_kg_kg * 1000
            P = b.pressure - 101325 * pyunits.Pa
            P_MPa = pyunits.convert(P, to_units=pyunits.MPa)

            h_w = (
                b.params.enth_mass_param_C1
                + b.params.enth_mass_param_C2 * t
                + b.params.enth_mass_param_C3 * t**2
                + b.params.enth_mass_param_C4 * t**3
            )
            h_sw0 = h_w - S_kg_kg * (
                b.params.enth_mass_param_B1
                + b.params.enth_mass_param_B2 * S_kg_kg
                + b.params.enth_mass_param_B3 * S_kg_kg**2
                + b.params.enth_mass_param_B4 * S_kg_kg**3
                + b.params.enth_mass_param_B5 * t
                + b.params.enth_mass_param_B6 * t**2
                + b.params.enth_mass_param_B7 * t**3
                + b.params.enth_mass_param_B8 * S_kg_kg * t
                + b.params.enth_mass_param_B9 * S_kg_kg**2 * t
                + b.params.enth_mass_param_B10 * S_kg_kg * t**2
            )
            h_sw = h_sw0 + P_MPa * (
                b.params.enth_mass_param_A1
                + b.params.enth_mass_param_A2 * t
                + b.params.enth_mass_param_A3 * t**2
                + b.params.enth_mass_param_A4 * t**3
                + S_g_kg
                * (
                    +b.params.enth_mass_param_A5
                    + b.params.enth_mass_param_A6 * t
                    + b.params.enth_mass_param_A7 * t**2
                    + b.params.enth_mass_param_A8 * t**3
                )
            )
            return b.enth_mass_phase[p] == h_sw

        self.eq_enth_mass_phase = Constraint(
            self.params.phase_list, rule=rule_enth_mass_phase
        )

    def _pressure_sat(self):
        params = self.params
        if not hasattr(params, "pressure_sat_param_psatw_A1"):
            t_inv_units = pyunits.K**-1
            s_inv_units = pyunits.kg / pyunits.g
            # vapor pressure parameters,  0-180 C, 0-160 g/kg
            # eq. 5 and 6 in Nayar et al.(2016)
            params.pressure_sat_param_psatw_A1 = Var(
                within=Reals,
                initialize=-5.8002206e3,
                units=pyunits.K,
                doc="Vapor pressure of pure water parameter A1",
            )
            params.pressure_sat_param_psatw_A2 = Var(
                within=Reals,
                initialize=1.3914993,
                units=pyunits.dimensionless,
                doc="Vapor pressure of pure water parameter A2",
            )
            params.pressure_sat_param_psatw_A3 = Var(
                within=Reals,
                initialize=-4.8640239e-2,
                units=t_inv_units,
                doc="Vapor pressure of pure water parameter A3",
            )
            params.pressure_sat_param_psatw_A4 = Var(
                within=Reals,
                initialize=4.1764768e-5,
                units=t_inv_units**2,
                doc="Vapor pressure of pure water parameter A4",
            )
            params.pressure_sat_param_psatw_A5 = Var(
                within=Reals,
                initialize=-1.4452093e-8,
                units=t_inv_units**3,
                doc="Vapor pressure of pure water parameter A5",
            )
            params.pressure_sat_param_psatw_A6 = Var(
                within=Reals,
                initialize=6.5459673,
                units=pyunits.dimensionless,
                doc="Vapor pressure of pure water parameter A6",
            )
            params.pressure_sat_param_B1 = Var(
                within=Reals,
                initialize=-4.5818e-4,
                units=s_inv_units,
                doc="Vapor pressure of seawater parameter B1",
            )
            params.pressure_sat_param_B2 = Var(
                within=Reals,
                initialize=-2.0443e-6,
                units=s_inv_units**2,
                doc="Vapor pressure of seawater parameter B2",
            )

            for v in params.component_objects(Var):
                v.fix()

        self.pressure_sat = Var(
            initialize=1e3,
            bounds=(1, 1e8),
            units=pyunits.Pa,
            doc="Saturation vapor pressure",
        )

        # Nayar et al.(2016), eq. 5 and 6, 0-180 C, 0-160 g/kg
        def rule_pressure_sat(b):
            t = b.temperature
            mass_solids = []
            for j in b.params.component_list:
                if j not in b.params.solvent_set:
                    mass_solids.append(b.flow_mass_phase_comp["Liq", j])
            mass_solution = mass_solids[:]
            mass_solution.append(b.flow_mass_phase_comp["Liq", "H2O"])
            S_kg_kg = sum(mass_solids) / sum(mass_solution)

            S_g_kg = S_kg_kg * 1000 * pyunits.g / pyunits.kg
            psatw = (
                exp(
                    b.params.pressure_sat_param_psatw_A1 * t**-1
                    + b.params.pressure_sat_param_psatw_A2
                    + b.params.pressure_sat_param_psatw_A3 * t
                    + b.params.pressure_sat_param_psatw_A4 * t**2
                    + b.params.pressure_sat_param_psatw_A5 * t**3
                    + b.params.pressure_sat_param_psatw_A6 * log(t / pyunits.K)
                )
                * pyunits.Pa
            )
            return b.pressure_sat == psatw * exp(
                b.params.pressure_sat_param_B1 * S_g_kg
                + b.params.pressure_sat_param_B2 * S_g_kg**2
            )

        self.eq_pressure_sat = Constraint(rule=rule_pressure_sat)

    def get_enthalpy_flow_terms(self, p):
        """Create enthalpy flow terms."""
        return self.enth_flow

    def default_energy_balance_type(self):
        return EnergyBalanceType.enthalpyTotal
