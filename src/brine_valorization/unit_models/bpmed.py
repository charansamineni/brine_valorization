from reaktoro_enabled_watertap.utils.watertap_flowsheet_block import (
    WaterTapFlowsheetBlockData,
)
from idaes.core.util.initialization import propagate_state
from watertap.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import (
    assert_optimal_termination,
)

from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
    ActivityCoefficientModel,
    DensityCalculation,
)
from idaes.core.util.math import smooth_min

from watertap.unit_models.pressure_changer import Pump
from watertap.costing.unit_models.pump import cost_pump
from pyomo.environ import (
    Var,
    value,
    Constraint,
    Objective,
    units as pyunits,
)
from pyomo.common.config import ConfigValue

from brine_valorization.property_models.mcas_with_enthalpy import (
    MCASWEParameterBlock,
)
from brine_valorization.unit_models.Biploar_and_Electrodialysis_1D_nmsu import (
    Bipolar_and_Electrodialysis1D,
    ElectricalOperationMode,
    LimitingCurrentDensitybpemMethod,
    PressureDropMethod,
    FrictionFactorMethod,
    HydraulicDiameterMethod,
)
from idaes.core import (
    declare_process_block_class,
)
from idaes.core import UnitModelCostingBlock

import idaes.core.util.scaling as iscale
from idaes.core.util.initialization import fix_state_vars, revert_state_vars
from idaes.models.unit_models import (
    Translator,
)
from reaktoro_enabled_watertap.utils import scale_utils as scu

from pyomo.network import Arc

from idaes.models.unit_models import (
    Separator,
    Mixer,
    Product,
    Feed,
    SplittingType,
    MomentumMixingType,
    MixingType,
    HeatExchangerFlowPattern,
)
from pyomo.util.calc_var_value import calculate_variable_from_constraint
try:
    from brackish_valorization_reaktoro.unit_models.multi_comp_mvc import MultiCompMVC
except ImportError:
    MultiCompMVC = None

__author__ = "Alexander V. Dudchenko"


@declare_process_block_class("BPMED")
class BPMEDData(WaterTapFlowsheetBlockData):
    CONFIG = WaterTapFlowsheetBlockData.CONFIG()
    CONFIG.declare(
        "bpmed_options_dict",
        ConfigValue(
            default=None,
            description="Options for bpmed, will override the defaults",
            doc="""
            Provide dict with options to change defaults in bpmed model,
            {'has_pressure_change:True} etc. 
            This will update default dictionary. 
            """,
        ),
    )
    CONFIG.declare(
        "bpmed_stages",
        ConfigValue(
            default=1,
            description="Number of stages for bpmed model",
            doc="""
            Provide the number of stages for the bpmed model.
            """,
        ),
    )

    CONFIG.declare(
        "track_pE",
        ConfigValue(
            default=False,
            description="if pE should be tracked in the model",
            doc="""
                    Providing True will add pE variable to the model and track it
            """,
        ),
    )
    CONFIG.declare(
        "target_recovery",
        ConfigValue(
            default=0.25,
            description="Target recovery for RO stage during initialization",
            doc="""
            Sets a target recovery for RO stage during initialization
            """,
        ),
    )
    CONFIG.declare(
        "bpmed_tracked_ions",
        ConfigValue(
            default=["H2O", "H_+", "OH_-", "Na_+", "Cl_-"],
            description="List of ions that should be passed into bpmed model",
            doc="""
            Providing a list of ions will include them in the bpmed model, other ions will
            be assumed to pass through directly through bpmed system
            """,
        ),
    )
    CONFIG.declare(
        "acid_products",
        ConfigValue(
            default={
                "HCl": {
                    "stoichiometric_reaction": {"H_+": 1, "Cl_-": 1},
                    "mass_tracking": {"H_+": 1},
                    "required_concentration_mass_basis": 0.32,
                    "cost": 0.12,
                },
            },
            description="Dictionary with expected products and their composition",
        ),
    )
    CONFIG.declare(
        "base_products",
        ConfigValue(
            default={
                "NaOH": {
                    "stoichiometric_reaction": {"OH_-": 1, "Na_+": 1},
                    "mass_tracking": {"OH_-": 1},
                    "required_concentration_mass_basis": 0.25,
                    "cost": 0.12,
                },
            },
            description="Dictionary with expected products and their composition",
        ),
    )
    CONFIG.declare(
        "add_feed_bleed_for_acid_base",
        ConfigValue(
            default=False,
            description="If feed and bleed configuration should be added",
        ),
    )
    CONFIG.declare(
        "add_feed_bleed_for_brine",
        ConfigValue(
            default=False,
            description="If feed and bleed configuration should be added",
        ),
    )
    CONFIG.declare(
        "add_mvc_concentrators",
        ConfigValue(
            default=False,
            description="If MVR concentrators should be added for acid and base product streams",
        ),
    )
    CONFIG.declare(
        "recycle_mvc_distillate",
        ConfigValue(
            default=False,
            description="If MVR distillate should be recycled back to the feed",
        ),
    )
    CONFIG.declare(
        "mvc_vapor_prop_pack",
        ConfigValue(
            default=None,
            description="Property package to use with MVC concentrators instead of ebdm property package",
        ),
    )
    CONFIG.declare(
        "mvc_feed_prop_pack",
        ConfigValue(
            default=None,
            description="Property package to use with MVC concentrators instead of ebdm property package",
        ),
    )
    CONFIG.declare(
        "enforce_equal_product_mass_flow",
        ConfigValue(
            default=True,
            description="If True, will enforce equal mass flow of acid and base products",
        ),
    )

    def build(self):
        super().build()
        self.create_bpmed_property_package()

        self.low_tds_splitter = Separator(
            property_package=self.bpmed_property_package,
            outlet_list=["acidate", "basate"],
            split_basis=SplittingType.totalFlow,
        )
        self.acidate_pump = Pump(
            property_package=self.bpmed_property_package,
        )
        self.basate_pump = Pump(
            property_package=self.bpmed_property_package,
        )
        self.brine_pump = Pump(
            property_package=self.config.default_property_package,
        )
        if self.default_property_package_diff_from_base:
            self.brine_translator_in = Translator(
                inlet_property_package=self.config.default_property_package,
                outlet_property_package=self.bpmed_property_package,
            )
            self.diluate_translator_out = Translator(
                inlet_property_package=self.bpmed_property_package,
                outlet_property_package=self.config.default_property_package,
            )
            self.dilute_water_translator_in = Translator(
                inlet_property_package=self.config.default_property_package,
                outlet_property_package=self.bpmed_property_package,
            )
        if (
            self.config.add_feed_bleed_for_acid_base
            or self.config.add_feed_bleed_for_brine
            or self.config.add_mvc_concentrators
        ):
            if self.config.add_feed_bleed_for_acid_base:
                recycle_splitter_outlets = ["recycle", "bleed"]
                self.acidate_splitter = Separator(
                    property_package=self.bpmed_property_package,
                    outlet_list=recycle_splitter_outlets,
                    split_basis=SplittingType.totalFlow,
                )
                self.basate_splitter = Separator(
                    property_package=self.bpmed_property_package,
                    outlet_list=recycle_splitter_outlets,
                    split_basis=SplittingType.totalFlow,
                )
            if self.config.add_feed_bleed_for_brine:
                recycle_splitter_outlets = ["recycle", "bleed"]
                self.diluate_splitter = Separator(
                    property_package=self.config.default_property_package,
                    outlet_list=recycle_splitter_outlets,
                    split_basis=SplittingType.totalFlow,
                )
            mixer_streams = ["feed"]
            if self.config.add_feed_bleed_for_acid_base:
                mixer_streams.append("recycle")
            if self.config.add_mvc_concentrators:
                self.acidate_mvc = MultiCompMVC(
                    default_property_package=self.bpmed_property_package,
                    mvc_property_package=self.config.mvc_feed_prop_pack,
                    add_reaktoro_chemistry=False,
                    vapor_property_package=self.config.mvc_vapor_prop_pack,
                    track_pH=False,
                    default_costing_package=self.config.default_costing_package,
                )
                self.basate_mvc = MultiCompMVC(
                    default_property_package=self.bpmed_property_package,
                    default_costing_package=self.config.default_costing_package,
                    mvc_property_package=self.config.mvc_feed_prop_pack,
                    add_reaktoro_chemistry=False,
                    vapor_property_package=self.config.mvc_vapor_prop_pack,
                    track_pH=False,
                )
                if self.config.recycle_mvc_distillate:

                    mixer_streams.append("mvr_distillate")
                    self.basate_distillate_pump = Pump(
                        property_package=self.bpmed_property_package,
                    )
                    self.acidate_distillate_pump = Pump(
                        property_package=self.bpmed_property_package,
                    )
                else:
                    self.distillate_mixer = Mixer(
                        property_package=self.bpmed_property_package,
                        inlet_list=["acidate_mvc", "basate_mvc"],
                        energy_mixing_type=MixingType.extensive,
                        momentum_mixing_type=MomentumMixingType.minimize,
                    )
        if (
            self.config.add_feed_bleed_for_acid_base
            or self.config.add_feed_bleed_for_brine
            or (
                self.config.recycle_mvc_distillate and self.config.add_mvc_concentrators
            )
        ):
            self.acidate_mixer = Mixer(
                property_package=self.bpmed_property_package,
                inlet_list=mixer_streams,
                energy_mixing_type=MixingType.none,
                momentum_mixing_type=MomentumMixingType.equality,
            )
            self.acidate_mixer.eq_temp = Constraint(
                expr=self.acidate_mixer.feed_state[0].temperature
                == self.acidate_mixer.mixed_state[0].temperature
            )
            self.basate_mixer = Mixer(
                property_package=self.bpmed_property_package,
                inlet_list=mixer_streams,
                energy_mixing_type=MixingType.none,
                momentum_mixing_type=MomentumMixingType.equality,
            )
            self.basate_mixer.eq_temp = Constraint(
                expr=self.basate_mixer.feed_state[0].temperature
                == self.basate_mixer.mixed_state[0].temperature
            )
            if self.config.add_feed_bleed_for_brine:
                self.brine_mixer = Mixer(
                    property_package=self.config.default_property_package,
                    inlet_list=["feed", "recycle"],
                    energy_mixing_type=MixingType.none,
                    momentum_mixing_type=MomentumMixingType.equality,
                )
                self.brine_mixer.eq_temp = Constraint(
                    expr=self.brine_mixer.feed_state[0].temperature
                    == self.brine_mixer.mixed_state[0].temperature
                )
                self.diluate_recycle_pump = Pump(
                    property_package=self.config.default_property_package,
                )
            if self.config.add_feed_bleed_for_acid_base:
                self.acidate_recycle_pump = Pump(
                    property_package=self.bpmed_property_package,
                )
                self.basate_recycle_pump = Pump(
                    property_package=self.bpmed_property_package,
                )
        self.bpmed_stages = list(range(self.config.bpmed_stages))
        self.bpmed = Bipolar_and_Electrodialysis1D(
            self.bpmed_stages,
            property_package=self.bpmed_property_package,
            has_pressure_change=True,
            pressure_drop_method=PressureDropMethod.Darcy_Weisbach,
            operation_mode=ElectricalOperationMode.Constant_Voltage,
            finite_elements=10,
            friction_factor_method=FrictionFactorMethod.Gurreri,
            hydraulic_diameter_method=HydraulicDiameterMethod.conventional,
            has_catalyst=True,
            salt_calculation=True,
            limiting_current_density_method_bpem=LimitingCurrentDensitybpemMethod.Empirical,
        )

        if self.default_property_package_diff_from_base:
            self.dilute_water_to_splitter = Arc(
                source=self.dilute_water_translator_in.outlet,
                destination=self.low_tds_splitter.inlet,
            )
            self.build_input_translator_connections(self.dilute_water_translator_in)
        self.splitter_to_acidate_pump = Arc(
            source=self.low_tds_splitter.acidate,
            destination=self.acidate_pump.inlet,
        )
        self.splitter_to_basate_pump = Arc(
            source=self.low_tds_splitter.basate,
            destination=self.basate_pump.inlet,
        )
        if self.config.add_feed_bleed_for_brine:
            self.brine_pump_to_brine_mixer = Arc(
                source=self.brine_pump.outlet,
                destination=self.brine_mixer.feed,
            )
        if self.default_property_package_diff_from_base:
            # build translator connections
            if self.config.add_feed_bleed_for_brine:
                self.brine_mixer_to_brine_translator = Arc(
                    source=self.brine_mixer.outlet,
                    destination=self.brine_translator_in.inlet,
                )
            else:
                self.brine_pump_to_brine_translator = Arc(
                    source=self.brine_pump.outlet,
                    destination=self.brine_translator_in.inlet,
                )

            self.brine_translator_to_bpmed = Arc(
                source=self.brine_translator_in.outlet,
                destination=self.bpmed[0].inlet_diluate,
            )

            # build all inputs
            self.build_input_translator_connections(self.brine_translator_in)
            self.build_output_translator_connections(self.diluate_translator_out)
            # build bypass constraints
            self.build_bypass_constraint(
                self.brine_translator_in,
                self.brine_translator_in.properties_in[0].flow_mol_phase_comp,
                self.diluate_translator_out.properties_out[0].flow_mol_phase_comp,
            )

        else:
            if self.config.add_feed_bleed_for_brine:
                self.brine_mixer_to_bpmed = Arc(
                    source=self.brine_mixer.outlet,
                    destination=self.bpmed[0].inlet_diluate,
                )
            else:
                self.brine_pump_to_bpmed = Arc(
                    source=self.brine_pump.outlet,
                    destination=self.bpmed[0].inlet_diluate,
                )
        if self.config.bpmed_stages > 1:
            for stage in self.bpmed_stages[:-1]:
                self.add_component(
                    f"sts_{stage}_diluate",
                    Arc(
                        source=self.bpmed[stage].outlet_diluate,
                        destination=self.bpmed[stage + 1].inlet_diluate,
                    ),
                )
                self.add_component(
                    f"sts_{stage}_acidate",
                    Arc(
                        source=self.bpmed[stage].outlet_acidate,
                        destination=self.bpmed[stage + 1].inlet_acidate,
                    ),
                )
                self.add_component(
                    f"sts_{stage}_basate",
                    Arc(
                        source=self.bpmed[stage].outlet_basate,
                        destination=self.bpmed[stage + 1].inlet_basate,
                    ),
                )
                assert self.find_component(f"sts_{stage}_diluate") is not None
                assert self.find_component(f"sts_{stage}_acidate") is not None
                assert self.find_component(f"sts_{stage}_basate") is not None
        if self.config.add_feed_bleed_for_acid_base or (
            self.config.add_mvc_concentrators and self.config.recycle_mvc_distillate
        ):
            self.acidate_pump_to_mixer = Arc(
                source=self.acidate_pump.outlet,
                destination=self.acidate_mixer.feed,
            )
            self.basate_pump_to_mixer = Arc(
                source=self.basate_pump.outlet,
                destination=self.basate_mixer.feed,
            )
            self.acidate_mixer_to_bpmed = Arc(
                source=self.acidate_mixer.outlet,
                destination=self.bpmed[0].inlet_acidate,
            )
            self.basate_mixer_to_bpmed = Arc(
                source=self.basate_mixer.outlet,
                destination=self.bpmed[0].inlet_basate,
            )

        if self.config.add_feed_bleed_for_acid_base:
            self.bpmed_to_acidate_splitter = Arc(
                source=self.bpmed[self.bpmed_stages[-1]].outlet_acidate,
                destination=self.acidate_splitter.inlet,
            )
            self.bpmed_to_basate_splitter = Arc(
                source=self.bpmed[self.bpmed_stages[-1]].outlet_basate,
                destination=self.basate_splitter.inlet,
            )
            self.acidate_splitter_to_recycle_pump = Arc(
                source=self.acidate_splitter.recycle,
                destination=self.acidate_recycle_pump.inlet,
            )
            self.acidate_recycle_pump_to_mixer = Arc(
                source=self.acidate_recycle_pump.outlet,
                destination=self.acidate_mixer.recycle,
            )
            self.basate_splitter_to_recycle_pump = Arc(
                source=self.basate_splitter.recycle,
                destination=self.basate_recycle_pump.inlet,
            )
            self.basate_recycle_pump_to_mixer = Arc(
                source=self.basate_recycle_pump.outlet,
                destination=self.basate_mixer.recycle,
            )

        if (
            self.config.add_feed_bleed_for_acid_base
            and self.config.add_mvc_concentrators
        ):
            self.acidate_splitter_to_acidate_mvc = Arc(
                source=self.acidate_splitter.bleed,
                destination=self.acidate_mvc.inlet.port,
            )
            self.basate_splitter_to_basate_mvc = Arc(
                source=self.basate_splitter.bleed,
                destination=self.basate_mvc.inlet.port,
            )
        elif self.config.add_mvc_concentrators:
            self.bpmed_to_acidate_mvc = Arc(
                source=self.bpmed[self.bpmed_stages[-1]].outlet_acidate,
                destination=self.acidate_mvc.inlet.port,
            )
            self.bpmed_to_basate_mvc = Arc(
                source=self.bpmed[self.bpmed_stages[-1]].outlet_basate,
                destination=self.basate_mvc.inlet.port,
            )
        if self.config.add_mvc_concentrators:
            if self.config.recycle_mvc_distillate:
                self.acidate_mvc.distillate.connect_to(
                    self.acidate_distillate_pump.inlet
                )

                self.basate_mvc.distillate.connect_to(self.basate_distillate_pump.inlet)

                self.acidate_distillate_pump_to_acidate_mixer = Arc(
                    source=self.acidate_distillate_pump.outlet,
                    destination=self.acidate_mixer.mvr_distillate,
                )
                self.basate_distillate_pump_to_basate_mixer = Arc(
                    source=self.basate_distillate_pump.outlet,
                    destination=self.basate_mixer.mvr_distillate,
                )
            else:
                self.acidate_mvc.distillate.connect_to(
                    self.distillate_mixer.acidate_mvc
                )
                self.basate_mvc.distillate.connect_to(self.distillate_mixer.basate_mvc)
        if not self.config.add_feed_bleed_for_acid_base and not (
            self.config.add_mvc_concentrators and self.config.recycle_mvc_distillate
        ):
            self.acidate_pump_to_bpmed = Arc(
                source=self.acidate_pump.outlet, destination=self.bpmed[0].inlet_acidate
            )
            self.basate_pump_to_bpmed = Arc(
                source=self.basate_pump.outlet,
                destination=self.bpmed[0].inlet_basate,
            )
        if self.default_property_package_diff_from_base:
            self.bpmed_diluate_to_translator_out = Arc(
                source=self.bpmed[self.bpmed_stages[-1]].outlet_diluate,
                destination=self.diluate_translator_out.inlet,
            )
        if self.config.add_feed_bleed_for_brine:
            if self.default_property_package_diff_from_base:
                self.diluate_translator_to_splitter = Arc(
                    source=self.diluate_translator_out.outlet,
                    destination=self.diluate_splitter.inlet,
                )
            else:
                self.bpmed_diluate_to_splitter = Arc(
                    source=self.bpmed[self.bpmed_stages[-1]].outlet_diluate,
                    destination=self.diluate_splitter.inlet,
                )
            self.diluate_splitter_to_recycle_pump = Arc(
                source=self.diluate_splitter.recycle,
                destination=self.diluate_recycle_pump.inlet,
            )
            self.diluate_recycle_pump_to_mixer = Arc(
                source=self.diluate_recycle_pump.outlet,
                destination=self.brine_mixer.recycle,
            )

        self.pH = Var(
            ["brine_inlet", "low_tds_water_inlet"],
            initialize=7,
            doc="pH of the feed to bpmed",
        )
        brine_vars = {"pH": self.pH["brine_inlet"]}
        di_vars = {"pH": self.pH["low_tds_water_inlet"]}
        if self.config.track_pE:
            self.pE = Var(
                ["brine_inlet", "low_tds_water_inlet"],
                initialize=0,
                doc="pE of the feed to bpmed",
            )
            brine_vars["pE"] = self.pE["brine_inlet"]
            di_vars["pE"] = self.pE["low_tds_water_inlet"]
        self.register_port(
            "brine_inlet",
            self.brine_pump.inlet,
            brine_vars,
        )

        if self.default_property_package_diff_from_base:
            self.register_port(
                "low_tds_water_inlet", self.dilute_water_translator_in.inlet, di_vars
            )
        else:
            self.register_port(
                "low_tds_water_inlet", self.low_tds_splitter.inlet, di_vars
            )
        if self.config.add_feed_bleed_for_brine:
            self.register_port(
                "diluate_outlet", self.diluate_splitter.bleed, brine_vars
            )

        else:
            if self.default_property_package_diff_from_base:
                self.register_port(
                    "diluate_outlet", self.diluate_translator_out.outlet, brine_vars
                )
            else:
                self.register_port(
                    "diluate_outlet",
                    self.bpmed[self.bpmed_stages[-1]].outlet_diluate,
                    brine_vars,
                )
        if self.config.add_mvc_concentrators:
            self.register_port("acidate_outlet", self.acidate_mvc.brine.port)
            self.register_port("basate_outlet", self.basate_mvc.brine.port)
            if self.config.recycle_mvc_distillate == False:
                self.register_port("mvc_distillate", self.distillate_mixer.outlet)
        elif self.config.add_feed_bleed_for_acid_base:
            self.register_port("acidate_outlet", self.acidate_splitter.bleed)
            self.register_port("basate_outlet", self.basate_splitter.bleed)
        else:
            self.register_port(
                "acidate_outlet", self.bpmed[self.bpmed_stages[-1]].outlet_acidate
            )
            self.register_port(
                "basate_outlet", self.bpmed[self.bpmed_stages[-1]].outlet_basate
            )
        self.add_expected_products()
        if self.config.default_costing_package is not None:
            self.add_costing()
        self.build_nacl_recovery_constraint()

    def add_costing(self):
        for stage in self.bpmed_stages:
            self.bpmed[stage].costing = UnitModelCostingBlock(
                flowsheet_costing_block=self.config.default_costing_package,
                costing_method_arguments={
                    "cost_electricity_flow": True,
                    "has_rectifier": True,
                },
            )
        self.acidate_pump.costing = UnitModelCostingBlock(
            flowsheet_costing_block=self.config.default_costing_package,
            costing_method_arguments={"pump_type": "low_pressure"},
        )
        self.basate_pump.costing = UnitModelCostingBlock(
            flowsheet_costing_block=self.config.default_costing_package,
            costing_method_arguments={"pump_type": "low_pressure"},
        )
        self.brine_pump.costing = UnitModelCostingBlock(
            flowsheet_costing_block=self.config.default_costing_package,
            costing_method_arguments={"pump_type": "low_pressure"},
        )
        if self.config.add_feed_bleed_for_acid_base:
            self.acidate_recycle_pump.costing = UnitModelCostingBlock(
                flowsheet_costing_block=self.config.default_costing_package,
                costing_method_arguments={"pump_type": "low_pressure"},
            )
            self.basate_recycle_pump.costing = UnitModelCostingBlock(
                flowsheet_costing_block=self.config.default_costing_package,
                costing_method_arguments={"pump_type": "low_pressure"},
            )
        if self.config.add_feed_bleed_for_brine:
            self.diluate_recycle_pump.costing = UnitModelCostingBlock(
                flowsheet_costing_block=self.config.default_costing_package,
                costing_method_arguments={"pump_type": "low_pressure"},
            )
        if self.config.add_mvc_concentrators and self.config.recycle_mvc_distillate:
            self.basate_distillate_pump.costing = UnitModelCostingBlock(
                flowsheet_costing_block=self.config.default_costing_package,
                costing_method_arguments={"pump_type": "low_pressure"},
            )
            self.acidate_distillate_pump.costing = UnitModelCostingBlock(
                flowsheet_costing_block=self.config.default_costing_package,
                costing_method_arguments={"pump_type": "low_pressure"},
            )

    def activate_product_quality_constraints(
        self, target_concentration=None, process=None
    ):
        if process is None:
            if self.config.add_mvc_concentrators:
                process = "mvc"
            else:
                process = "bpmed"
        if target_concentration is not None:
            for p, product in self.product_target_concentration:
                if p == process:
                    if isinstance(target_concentration, dict):
                        if product in target_concentration:
                            self.product_target_concentration[p, product].fix(
                                target_concentration[product]
                            )
                    else:
                        self.product_target_concentration[p, product].fix(
                            target_concentration
                        )
                    self.eq_product_target_concentration[p, product].activate()
        self.eq_product_target_concentration.display()
        self.product_target_concentration.display()

    def deactivate_product_quality_constraints(self):
        self.eq_product_target_concentration.deactivate()

    def add_expected_products(self):
        process = ["bpmed"]
        if self.config.add_mvc_concentrators:
            process.append("mvc")
        self.process = process
        self.flow_mass_product = Var(
            process,
            list(self.config.acid_products.keys())
            + list(self.config.base_products.keys()),
            initialize=0,
            units=pyunits.kg / pyunits.s,
            doc="Mass flow of expected products",
        )
        self.product_mass_concentration = Var(
            process,
            list(self.config.acid_products.keys())
            + list(self.config.base_products.keys()),
            initialize=0,
            units=pyunits.kg / pyunits.kg,
            doc="Mass concentration of expected products",
        )
        self.product_target_concentration = Var(
            process,
            list(self.config.acid_products.keys())
            + list(self.config.base_products.keys()),
            initialize=0,
            units=pyunits.kg / pyunits.kg,
            doc="Target mass concentration of expected products",
        )
        for product, product_data in self.config.acid_products.items():
            for p in process:
                self.product_target_concentration[p, product].fix(
                    product_data["required_concentration_mass_basis"]
                )
        for product, product_data in self.config.base_products.items():
            for p in process:
                self.product_target_concentration[p, product].fix(
                    product_data["required_concentration_mass_basis"]
                )

        def get_acid_stream(process="bpmed"):
            if self.config.add_mvc_concentrators and process == "mvc":
                return self.acidate_mvc.get_brine_state()
            elif self.config.add_feed_bleed_for_acid_base and process == "bpmed":
                return self.acidate_splitter.bleed_state[0]
            else:
                return self.bpmed[self.bpmed_stages[-1]].acidate.properties[0, 1]

        def get_base_stream(process="bpmed"):
            if self.config.add_mvc_concentrators and process == "mvc":
                return self.basate_mvc.get_brine_state()
            elif self.config.add_feed_bleed_for_acid_base and process == "bpmed":
                return self.basate_splitter.bleed_state[0]
            else:
                return self.bpmed[self.bpmed_stages[-1]].basate.properties[0, 1]

        @self.Constraint(
            process,
            list(self.config.acid_products.keys())
            + list(self.config.base_products.keys()),
        )
        def eq_product_mass(b, p, product):
            mw = []
            if product in self.config.acid_products:
                product_stream = get_acid_stream(process=p).flow_mol_phase_comp
                stoich_reaction = self.config.acid_products[product][
                    "stoichiometric_reaction"
                ]
                for ion, coeff in stoich_reaction.items():
                    mw.append(self.bpmed_property_package.mw_comp[ion] * coeff)
                if "mass_tracking" in self.config.acid_products[product]:
                    stoich_reaction = self.config.acid_products[product][
                        "mass_tracking"
                    ]
            if product in self.config.base_products:
                product_stream = get_base_stream(process=p).flow_mol_phase_comp

                stoich_reaction = self.config.base_products[product][
                    "stoichiometric_reaction"
                ]
                for ion, coeff in stoich_reaction.items():
                    mw.append(self.bpmed_property_package.mw_comp[ion] * coeff)
                if "mass_tracking" in self.config.base_products[product]:
                    stoich_reaction = self.config.base_products[product][
                        "mass_tracking"
                    ]

            ssmoth_min = []
            for ion, coeff in stoich_reaction.items():
                if coeff > 0:
                    ssmoth_min.append(product_stream["Liq", ion] * coeff)
            if len(ssmoth_min) == 1:
                e_ssmooth_min = ssmoth_min[0]
            else:
                e_ssmooth_min = smooth_min(ssmoth_min[0], ssmoth_min[1], eps=1e-12)
                for i in ssmoth_min[2:]:
                    e_ssmooth_min = smooth_min(e_ssmooth_min[-1], i, eps=1e-12)
            mw = sum(mw)

            return b.flow_mass_product[p, product] == e_ssmooth_min * (mw)

        @self.Constraint(
            process,
            list(self.config.acid_products.keys())
            + list(self.config.base_products.keys()),
        )
        def eq_product_concentration(b, p, product):
            if product in self.config.acid_products:
                product_stream = get_acid_stream(process=p).flow_mass_phase_comp
            if product in self.config.base_products:
                product_stream = get_base_stream(process=p).flow_mass_phase_comp
            return b.product_mass_concentration[p, product] == self.flow_mass_product[
                p, product
            ] / sum(product_stream[p] for p in product_stream)

        @self.Constraint(
            process,
            list(self.config.acid_products.keys())
            + list(self.config.base_products.keys()),
        )
        def eq_product_target_concentration(b, p, product):
            return (
                b.product_mass_concentration[p, product]
                == b.product_target_concentration[p, product]
            )

        self.eq_product_target_concentration.deactivate()
        if self.config.enforce_equal_product_mass_flow:
            self.eq_equal_product_mass_flow = Constraint(
                expr=sum(
                    self.bpmed[self.bpmed_stages[-1]]
                    .acidate.properties[0, 1]
                    .flow_mass_phase_comp[ion]
                    for ion in self.bpmed[self.bpmed_stages[-1]]
                    .acidate.properties[0, 1]
                    .flow_mass_phase_comp
                )
                == sum(
                    self.bpmed[self.bpmed_stages[-1]]
                    .basate.properties[0, 1]
                    .flow_mass_phase_comp[ion]
                    for ion in self.bpmed[self.bpmed_stages[-1]]
                    .basate.properties[0, 1]
                    .flow_mass_phase_comp
                )
            )
            self.eq_equal_product_mass_flow.deactivate()

    def activate_equal_product_mass_flow_constraint(self):

        if self.config.enforce_equal_product_mass_flow:
            self.eq_equal_product_mass_flow.activate()
            self.low_tds_splitter.split_fraction[0, "acidate"].unfix()

    def build_input_translator_connections(self, translator_block):

        @translator_block.Constraint(
            list(translator_block.properties_in[0].flow_mol_phase_comp)
        )
        def eq_mol_flow(b, p, i):
            if i not in self.config.bpmed_tracked_ions:
                return Constraint.Skip

            else:
                return (
                    translator_block.properties_out[0].flow_mol_phase_comp[p, i]
                    == translator_block.properties_in[0].flow_mol_phase_comp[p, i]
                )

        @translator_block.Constraint(["H_+", "OH_-"])
        def eq_acid_flow(b, i):
            translator_block.properties_out[0].flow_mol_phase_comp["Liq", i].fix(1e-8)
            return (
                Constraint.Skip
            )  # translator_block.properties_out[0].flow_mol_phase_comp["Liq", i] == 0

        translator_block.eq_pressure_equality = Constraint(
            expr=translator_block.properties_in[0].pressure
            == translator_block.properties_out[0].pressure
        )
        translator_block.eq_temperature_equality = Constraint(
            expr=translator_block.properties_in[0].temperature
            == translator_block.properties_out[0].temperature
        )

    def build_output_translator_connections(self, translator_block):

        @translator_block.Constraint(
            list(translator_block.properties_out[0].flow_mol_phase_comp)
        )
        def eq_mol_flow(b, p, i):
            if i not in self.config.bpmed_tracked_ions:
                return Constraint.Skip
            else:
                return (
                    translator_block.properties_out[0].flow_mol_phase_comp[p, i]
                    == translator_block.properties_in[0].flow_mol_phase_comp[p, i]
                )

        translator_block.eq_pressure_equality = Constraint(
            expr=translator_block.properties_in[0].pressure
            == translator_block.properties_out[0].pressure
        )
        translator_block.eq_temperature_equality = Constraint(
            expr=translator_block.properties_in[0].temperature
            == translator_block.properties_out[0].temperature
        )

    def build_bypass_constraint(self, blk, inlet, outlet):

        @blk.Constraint(list(inlet))
        def eq_bypass(b, p, i):
            if i not in self.config.bpmed_tracked_ions:
                return outlet[p, i] == inlet[p, i]
            else:
                return Constraint.Skip

    def build_nacl_recovery_constraint(self):
        self.nacl_recovery = Var(
            initialize=0.5,
            units=pyunits.dimensionless,
            doc="Recovery of NaCl in the system",
        )
        iscale.set_scaling_factor(self.nacl_recovery, 1)
        if self.config.add_feed_bleed_for_brine:
            inlet = self.brine_mixer.feed_state[0]
            outlet = self.diluate_splitter.bleed_state[0]
        else:
            inlet = self.bpmed[self.bpmed_stages[0]].diluate.properties[0, 0]
            outlet = self.bpmed[self.bpmed_stages[-1]].diluate.properties[0, 1]
        self.eq_nacl_recovery = Constraint(
            expr=self.nacl_recovery
            == (
                (
                    inlet.flow_mass_phase_comp["Liq", "Na_+"]
                    + inlet.flow_mass_phase_comp["Liq", "Cl_-"]
                )
                - (
                    outlet.flow_mass_phase_comp["Liq", "Na_+"]
                    + outlet.flow_mass_phase_comp["Liq", "Cl_-"]
                )
            )
            / (
                inlet.flow_mass_phase_comp["Liq", "Na_+"]
                + inlet.flow_mass_phase_comp["Liq", "Cl_-"]
            )
        )

        iscale.set_scaling_factor(self.eq_nacl_recovery, 1)

    def create_bpmed_property_package(self):
        default_package_config = self.config.default_property_package.config
        mcas_props = {
            "solute_list": ["Na_+", "Cl_-", "H_+", "OH_-"],
            "mw_data": {
                "H2O": 18e-3,
                "Na_+": 23e-3,
                "Cl_-": 35.5e-3,
                "H_+": 1e-3,
                "OH_-": 17.0e-3,
            },
            "elec_mobility_data": {
                ("Liq", "Na_+"): 5.19e-8,
                ("Liq", "Cl_-"): 7.92e-8,
                ("Liq", "H_+"): 36.23e-8,
                ("Liq", "OH_-"): 20.64e-8,
            },
            "charge": {"Na_+": 1, "Cl_-": -1, "H_+": 1, "OH_-": -1},
            "diffusivity_data": {
                ("Liq", "Na_+"): 1.33e-9,
                ("Liq", "Cl_-"): 2.03e-9,
                ("Liq", "H_+"): 9.31e-9,
                ("Liq", "OH_-"): 5.27e-9,
            },
        }
        self.default_property_package_diff_from_base = False
        for ion in default_package_config.solute_list:
            if ion in self.config.bpmed_tracked_ions:
                if ion not in mcas_props["solute_list"]:
                    mcas_props["solute_list"].append(ion)
                    mcas_props["mw_data"][ion] = default_package_config.mw_comp[ion]
                    mcas_props["charge"][ion] = default_package_config.charge_comp[ion]
                    mcas_props["diffusivity_data"][("Liq", ion)] = (
                        default_package_config.diffusivity_data[("Liq", ion)]
                    )
                    mcas_props["elec_mobility_data"][("Liq", ion)] = (
                        default_package_config.elec_mobility_data[("Liq", ion)]
                    )
            else:
                self.default_property_package_diff_from_base = True
        if "H_+" not in default_package_config.solute_list:
            self.default_property_package_diff_from_base = True
        if "OH_-" not in default_package_config.solute_list:
            self.default_property_package_diff_from_base = True
        mcas_props["activity_coefficient_model"] = ActivityCoefficientModel.ideal
        mcas_props["density_calculation"] = DensityCalculation.constant
        self.bpmed_property_package = MCASWEParameterBlock(**mcas_props)

    def set_fixed_operation(self):
        self.low_tds_splitter.split_fraction[0, "acidate"].fix(0.5)
        self.low_tds_splitter.split_fraction[0, "acidate"].setlb(0.1)
        self.low_tds_splitter.split_fraction[0, "acidate"].setub(0.9)
        if self.config.add_feed_bleed_for_acid_base:
            self.acidate_splitter.split_fraction[0, "recycle"].fix(0.1)
            self.basate_splitter.split_fraction[0, "recycle"].fix(0.1)
            self.acidate_splitter.split_fraction[0, "recycle"].setlb(0.1)
            self.basate_splitter.split_fraction[0, "recycle"].setlb(0.1)
            self.acidate_splitter.split_fraction[0, "recycle"].setub(1 - 0.1)
            self.basate_splitter.split_fraction[0, "recycle"].setub(1 - 0.1)
            self.acidate_recycle_pump.efficiency_pump[0].fix(0.8)
            self.basate_recycle_pump.efficiency_pump[0].fix(0.8)
        if self.config.add_feed_bleed_for_brine:
            self.diluate_splitter.split_fraction[0, "recycle"].fix(0.1)
            self.diluate_splitter.split_fraction[0, "recycle"].setlb(0.1)
            self.diluate_splitter.split_fraction[0, "recycle"].setub(1 - 0.1)
            self.diluate_recycle_pump.efficiency_pump[0].fix(0.8)
        if self.config.add_mvc_concentrators:
            self.acidate_mvc.set_fixed_operation()
            self.basate_mvc.set_fixed_operation()
            if self.config.recycle_mvc_distillate:
                self.acidate_distillate_pump.efficiency_pump[0].fix(0.8)
                self.basate_distillate_pump.efficiency_pump[0].fix(0.8)

        self.acidate_pump.deltaP[0] = 3e5
        self.basate_pump.deltaP[0] = 3e5
        self.brine_pump.deltaP[0] = 3e5

        self.acidate_pump.efficiency_pump[0].fix(0.8)
        self.basate_pump.efficiency_pump[0].fix(0.8)
        self.brine_pump.efficiency_pump[0].fix(0.8)

        self.bpmed[self.bpmed_stages[-1]].outlet_diluate.pressure[0].fix(101325)
        self.bpmed[self.bpmed_stages[-1]].outlet_acidate.pressure[0].fix(101325)
        self.bpmed[self.bpmed_stages[-1]].outlet_basate.pressure[0].fix(101325)
        for _, bpmed in self.bpmed.items():

            bpmed.cell_triplet_num.fix(50)

            bpmed.cell_triplet_num.setlb(50)
            bpmed.electrical_stage_num.fix(1)
            # single pair sizing
            bpmed.cell_length.fix(1)
            bpmed.cell_width.fix(1)
            bpmed.cell_length.setlb(0.1)
            bpmed.cell_width.setlb(0.1)
            bpmed.channel_height["diluate"].fix(0.00038)
            bpmed.channel_height["basate"].fix(0.00038)
            bpmed.channel_height["acidate"].fix(0.00038)
            bpmed.channel_height["basate"].setlb(0.00015)
            bpmed.channel_height["acidate"].setlb(0.00015)
            bpmed.velocity_diluate[0, 0].setlb(0.05)
            bpmed.velocity_acidate[0, 0].setlb(0.01)
            bpmed.velocity_basate[0, 0].setlb(0.01)
            bpmed.velocity_diluate[0, 0].setub(0.25)
            bpmed.velocity_acidate[0, 0].setub(0.25)
            bpmed.velocity_basate[0, 0].setub(0.25)

            bpmed.channel_height["diluate"].fix(0.00038)
            if bpmed.find_component("equal_acidate_velocity") is None:
                bpmed.equal_acidate_velocity = Constraint(
                    expr=bpmed.velocity_acidate[0, 0] == bpmed.velocity_diluate[0, 0]
                )
                bpmed.equal_basate_velocity = Constraint(
                    expr=bpmed.velocity_basate[0, 0] == bpmed.velocity_diluate[0, 0]
                )
                iscale.constraint_scaling_transform(
                    bpmed.equal_acidate_velocity, 1 / 0.0001
                )
                iscale.constraint_scaling_transform(
                    bpmed.equal_basate_velocity, 1 / 0.0001
                )
                bpmed.channel_height["basate"].unfix()
                bpmed.channel_height["acidate"].unfix()

            bpmed.spacer_porosity.fix(0.9)
            bpmed.shadow_factor.fix(1)
            bpmed.membrane_thickness["aem"].fix(570.0e-6 * pyunits.m)
            bpmed.membrane_thickness["cem"].fix(570.0e-6 * pyunits.m)
            bpmed.membrane_thickness["bpem"].fix(2 * 570.0e-6 * pyunits.m)
            bpmed.diffus_mass.fix(1.6e-9)
            # membrane transport properties
            transport_props = {
                "aem": 0.97,
                "cem": 0.96,
                "cem_+1_diffusivity": 2.00e-10,
                "aem_+1_diffusivity": 7.50e-11,
                "cem_-1_diffusivity": 1.50e-10,
                "aem_-1_diffusivity": 1.90e-10,
                "cem_water": 5.8,
                "aem_water": 4.3,
            }
            for ion in self.bpmed_property_package.config.solute_list:
                ion_charge = self.bpmed_property_package.charge_comp[ion].value
                if ion_charge == 1:
                    bpmed.ion_trans_number_membrane["aem", ion].fix(
                        1 - transport_props["aem"]
                    )
                    bpmed.ion_trans_number_membrane["cem", ion].fix(
                        transport_props["cem"]
                    )
                if ion_charge == -1:
                    bpmed.ion_trans_number_membrane["aem", ion].fix(
                        transport_props["aem"]
                    )
                    bpmed.ion_trans_number_membrane["cem", ion].fix(
                        1 - transport_props["cem"]
                    )

            bpmed.ion_trans_number_membrane["aem", "H_+"].fix(0)
            bpmed.ion_trans_number_membrane["aem", "OH_-"].fix(0)
            bpmed.ion_trans_number_membrane["cem", "H_+"].fix(0)
            bpmed.ion_trans_number_membrane["cem", "OH_-"].fix(0)

            for ion in self.bpmed_property_package.config.solute_list:
                bpmed.ion_trans_number_membrane["bpem", ion].fix(0)

            bpmed.water_trans_number_membrane["cem"].fix(transport_props["cem_water"])
            bpmed.water_trans_number_membrane["aem"].fix(transport_props["aem_water"])
            bpmed.water_trans_number_membrane["bpem"].fix(
                (transport_props["cem_water"] + transport_props["aem_water"]) / 2
            )

            for ion in self.bpmed_property_package.config.solute_list:
                bpmed.ion_trans_number_membrane["bpem", ion].fix(0)

            bpmed.ion_trans_number_membrane["bpem", "H_+"].fix(1)
            bpmed.ion_trans_number_membrane["bpem", "OH_-"].fix(1)

            for ion in self.bpmed_property_package.config.solute_list:
                ion_charge = self.bpmed_property_package.charge_comp[ion].value
                if ion_charge == 1:
                    bpmed.solute_diffusivity_membrane["cem", ion].fix(
                        transport_props["cem_+1_diffusivity"]
                    )
                    bpmed.solute_diffusivity_membrane["aem", ion].fix(
                        transport_props["aem_+1_diffusivity"]
                    )
                if ion_charge == -1:
                    bpmed.solute_diffusivity_membrane["cem", ion].fix(
                        transport_props["cem_-1_diffusivity"]
                    )
                    bpmed.solute_diffusivity_membrane["aem", ion].fix(
                        transport_props["aem_-1_diffusivity"]
                    )
            for ion in self.bpmed_property_package.config.solute_list:
                bpmed.solute_diffusivity_membrane["bpem", ion].fix(0)
            bpmed.solute_diffusivity_membrane["cem", "H_+"].fix(0)
            bpmed.solute_diffusivity_membrane["aem", "H_+"].fix(0)
            bpmed.solute_diffusivity_membrane["cem", "OH_-"].fix(0)
            bpmed.solute_diffusivity_membrane["aem", "OH_-"].fix(0)
            bpmed.water_permeability_membrane["bpem"].fix((2.16e-14 + 1.75e-14) / 2)
            bpmed.water_permeability_membrane["cem"].fix(2.16e-14)
            bpmed.water_permeability_membrane["aem"].fix(1.75e-14)

            bpmed.current_utilization.fix(1)
            bpmed.electrodes_resistance.fix(0.01)
            bpmed.voltage_applied[0].fix(150)
            bpmed.voltage_applied.setub(800)
            bpmed.voltage_applied.setlb(50)
            bpmed.membrane_fixed_charge.fix(5e3)
            bpmed.conc_water.fix(50 * 1e3)
            bpmed.kr.fix(1.3 * 10**10)
            bpmed.k2_zero.fix(2 * 10**-6)
            bpmed.relative_permittivity.fix(30)

            bpmed.membrane_fixed_catalyst_cem.fix(5e3)
            bpmed.membrane_fixed_catalyst_aem.fix(5e3)
            bpmed.k_a.fix(1e1)
            bpmed.k_b.fix(5e4)
        self.nacl_recovery.unfix()

    def set_optimization_operation(self):
        if self.config.add_mvc_concentrators:  # and self.config.recycle_mvc_distillate:
            self.acidate_mvc.set_optimization_operation()
            self.basate_mvc.set_optimization_operation()
        self.acidate_pump.deltaP.unfix()
        self.basate_pump.deltaP.unfix()
        self.brine_pump.deltaP.unfix()
        self.acidate_pump.deltaP.setub(25 * pyunits.bar)
        self.basate_pump.deltaP.setub(25 * pyunits.bar)
        self.brine_pump.deltaP.setub(25 * pyunits.bar)
        self.bpmed[self.bpmed_stages[-1]].outlet_diluate.pressure[0].fix(101325)
        self.bpmed[self.bpmed_stages[-1]].outlet_acidate.pressure[0].fix(101325)
        self.bpmed[self.bpmed_stages[-1]].outlet_basate.pressure[0].fix(101325)
        # self.low_tds_splitter.split_fraction[0, "acidate"].unfix()

        for _, bpmed in self.bpmed.items():
            bpmed.electrical_stage_num.unfix()
            bpmed.cell_length.setlb(0.1)
            bpmed.cell_width.setlb(0.1)
            bpmed.cell_triplet_num.unfix()
            bpmed.cell_triplet_num.setlb(25)
            bpmed.cell_triplet_num.setub(1000)
            bpmed.electrical_stage_num.fix()
            bpmed.voltage_applied.unfix()
            bpmed.voltage_applied.setub(800)
            bpmed.voltage_applied.setlb(50)
            bpmed.cell_length.unfix()
            bpmed.cell_length.setub(100)
            bpmed.cell_width.unfix()
            bpmed.velocity_acidate[0, 0].setlb(0.01)
            bpmed.velocity_basate[0, 0].setlb(0.01)
            bpmed.velocity_diluate[0, 0].setlb(0.05)
            bpmed.velocity_diluate[0, 0].setub(0.5)
            bpmed.velocity_acidate[0, 0].setub(0.5)
            bpmed.velocity_basate[0, 0].setub(0.5)
        self.nacl_recovery.unfix()
        self.low_tds_splitter.split_fraction[0, "acidate"].unfix()
        if self.config.add_feed_bleed_for_acid_base:

            self.acidate_splitter.split_fraction[0, "recycle"].unfix()
            self.basate_splitter.split_fraction[0, "recycle"].unfix()
            self.acidate_splitter.split_fraction[0, "recycle"].setlb(0.01)
            self.basate_splitter.split_fraction[0, "recycle"].setlb(0.01)
            self.acidate_splitter.split_fraction[0, "recycle"].setub(0.99)
            self.basate_splitter.split_fraction[0, "recycle"].setub(0.99)

        if self.config.add_feed_bleed_for_brine:
            self.diluate_splitter.split_fraction[0, "recycle"].unfix()
            self.diluate_splitter.split_fraction[0, "recycle"].setlb(0.01)
            self.diluate_splitter.split_fraction[0, "recycle"].setub(0.99)

    def scale_before_initialization(self, flow_factor=1, solid_gen_factor=1, **kwargs):
        h2o_scale_factor = (
            self.config.default_property_package._default_scaling_factors[
                "flow_mol_phase_comp", ("Liq", "H2O")
            ]
        )
        for ion in self.bpmed_property_package.component_list:
            if ion in self.config.default_property_package.solute_set:
                scale_factor = (
                    self.config.default_property_package._default_scaling_factors[
                        "flow_mol_phase_comp", ("Liq", ion)
                    ]
                )

                self.bpmed_property_package.set_default_scaling(
                    "flow_mol_phase_comp", scale_factor, index=("Liq", ion)
                )
        self.bpmed_property_package.set_default_scaling(
            "flow_mol_phase_comp", h2o_scale_factor, index=("Liq", "H2O")
        )
        self.bpmed_property_package.set_default_scaling(
            "flow_mol_phase_comp", h2o_scale_factor * 10, index=("Liq", "H_+")
        )
        self.bpmed_property_package.set_default_scaling(
            "flow_mol_phase_comp", h2o_scale_factor * 10, index=("Liq", "OH_-")
        )
        fluid_flow_scale = (
            self.config.default_property_package._default_scaling_factors[
                "flow_mol_phase_comp", ("Liq", "H2O")
            ]
            / value(self.config.default_property_package.mw_comp["H2O"])
        )
        work_scale = 1e-3 / (1 / fluid_flow_scale)
        heat_scale = 1e-6 / (1 / fluid_flow_scale)

        for p, product in self.flow_mass_product:
            if product in self.config.base_products:
                stoich_reaction = self.config.base_products[product][
                    "stoichiometric_reaction"
                ]
            if product in self.config.acid_products:
                stoich_reaction = self.config.acid_products[product][
                    "stoichiometric_reaction"
                ]
            csfs = 0
            for ion, coeff in stoich_reaction.items():

                if coeff > 0:
                    sf = self.bpmed_property_package._default_scaling_factors[
                        "flow_mol_phase_comp", ("Liq", ion)
                    ]
                    csfs += (
                        1 / sf * value(self.bpmed_property_package.mw_comp[ion]) * coeff
                    )
            csfs = 10 / csfs
            iscale.set_scaling_factor(self.flow_mass_product[p, product], csfs)
            iscale.constraint_scaling_transform(self.eq_product_mass[p, product], csfs)

            iscale.set_scaling_factor(self.product_mass_concentration[p, product], 10)
            iscale.constraint_scaling_transform(
                self.eq_product_concentration[p, product], 10
            )
            iscale.set_scaling_factor(self.product_target_concentration[p, product], 10)
            iscale.set_scaling_factor(
                self.eq_product_target_concentration[p, product], 10
            )
            print(f"Scaling factor for {p},  {product} mass flow is {csfs}")

        if self.default_property_package_diff_from_base:
            for block, scale_type in [
                (self.brine_translator_in, "mol"),
                (self.dilute_water_translator_in, "mol"),
                (self.diluate_translator_out, "mol"),
            ]:
                iscale.constraint_scaling_transform(block.eq_pressure_equality, 1e-5)
                iscale.constraint_scaling_transform(block.eq_temperature_equality, 1e-2)

                # Scale the mass flow of the solute in the translator block,
                # for inlet, we use scale from ro_prop_scaling, for outlet we use default scaling
                # for default property package, Water scaling is same for both

                for phase, ion in block.eq_mol_flow:
                    sf = self.config.default_property_package._default_scaling_factors[
                        "flow_mol_phase_comp", ("Liq", ion)
                    ]
                    iscale.constraint_scaling_transform(
                        block.eq_mol_flow[phase, ion],
                        sf,
                    )
                if block.find_component("eq_acid_flow") is not None:
                    for ion in block.eq_acid_flow:
                        sf = self.bpmed_property_package._default_scaling_factors[
                            "flow_mol_phase_comp", ("Liq", ion)
                        ]
                        iscale.constraint_scaling_transform(block.eq_acid_flow[ion], sf)
        iscale.set_scaling_factor(self.acidate_pump.control_volume.work, work_scale)
        iscale.set_scaling_factor(self.basate_pump.control_volume.work, work_scale)
        iscale.set_scaling_factor(self.brine_pump.control_volume.work, work_scale)
        for _, bpmed in self.bpmed.items():
            iscale.set_scaling_factor(bpmed.diluate.area, 1)
            iscale.set_scaling_factor(bpmed.basate.area, 1)
            iscale.set_scaling_factor(bpmed.acidate.area, 1)

            iscale.set_scaling_factor(bpmed.cell_width, 10)
            iscale.set_scaling_factor(bpmed.cell_length, 10)
            var_to_scale = [
                bpmed.solute_diffusivity_membrane,
                bpmed.membrane_thickness,
                bpmed.water_permeability_membrane,
                bpmed.electrical_stage_num,
                bpmed.shadow_factor,
                bpmed.channel_height,
                bpmed.spacer_porosity,
                bpmed.voltage_applied,
                bpmed.conc_water,
                bpmed.membrane_fixed_catalyst_cem,
                bpmed.membrane_fixed_catalyst_aem,
                bpmed.elec_field_non_dim,
                bpmed.membrane_fixed_charge,
                bpmed.diffus_mass,
                bpmed.salt_conc_aem_x,
                bpmed.salt_conc_cem_x,
                bpmed.salt_conc_dilu_x,
                bpmed.relative_permittivity,
                bpmed.kr,
                bpmed.k2_zero,
                bpmed.voltage_membrane_drop,
            ]
            for var in var_to_scale:
                if var.is_indexed:
                    for index in var:
                        if value(var[index]) != 0:
                            iscale.set_scaling_factor(var[index], 1 / value(var[index]))
                        else:
                            print(
                                f"Warning: variable {var.name} with index {index} has value 0, cannot set scaling factor based on value."
                            )
                            iscale.set_scaling_factor(var[index], 1)
                else:
                    if value(var[index]) != 0:
                        iscale.set_scaling_factor(var, 1 / var.value)
                    else:
                        print(
                            f"Warning: variable {var.name} with index {index} has value 0, cannot set scaling factor based on value."
                        )
                        iscale.set_scaling_factor(var[index], 1)
            iscale.set_scaling_factor(bpmed.pressure_drop, 1e-5)
            iscale.set_scaling_factor(bpmed.pressure_drop_total, 1e-5)
            iscale.set_scaling_factor(bpmed.N_Re, 1e-2)
            iscale.set_scaling_factor(bpmed.N_Sh, 1)
            iscale.set_scaling_factor(bpmed.N_Sc, 1)
            iscale.set_scaling_factor(bpmed.velocity_diluate, 1e2)
            iscale.set_scaling_factor(bpmed.velocity_acidate, 1e2)
            iscale.set_scaling_factor(bpmed.velocity_basate, 1e2)
            iscale.set_scaling_factor(bpmed.friction_factor, 1e-2)
            iscale.set_scaling_factor(bpmed.hydraulic_diameter, 1e3)
            iscale.set_scaling_factor(bpmed.k_a, 1e-2)
            iscale.set_scaling_factor(bpmed.k_b, 1e-3)
            iscale.set_scaling_factor(bpmed.flux_splitting, 1e4)
            iscale.set_scaling_factor(bpmed.current_density_x, 1e-3)
            iscale.set_scaling_factor(bpmed.voltage_x, 1e-2)
            iscale.set_scaling_factor(bpmed.cell_triplet_num, 1 / 100)
            iscale.set_scaling_factor(bpmed.electrodes_resistance, 1e2)

        if self.config.add_mvc_concentrators:
            self.acidate_mvc.scale_before_initialization()
            self.basate_mvc.scale_before_initialization()
            if self.config.recycle_mvc_distillate:
                iscale.set_scaling_factor(
                    self.basate_distillate_pump.control_volume.work, work_scale
                )
                iscale.set_scaling_factor(
                    self.acidate_distillate_pump.control_volume.work, work_scale
                )
        if self.config.add_feed_bleed_for_brine:
            iscale.set_scaling_factor(
                self.diluate_recycle_pump.control_volume.work, work_scale
            )

        if self.config.add_feed_bleed_for_acid_base:
            iscale.set_scaling_factor(
                self.acidate_recycle_pump.control_volume.work, work_scale
            )
            iscale.set_scaling_factor(
                self.basate_recycle_pump.control_volume.work, work_scale
            )
        if self.config.default_costing_package is not None:
            cp = self.config.default_costing_package
            iscale.set_scaling_factor(
                cp.bipolar_electrodialysis_costing.stack_electrode_capital_cost,
                1 / 2100,
            )
            iscale.set_scaling_factor(
                cp.bipolar_electrodialysis_costing.membrane_capital_cost,
                1 / 100,
            )
            iscale.set_scaling_factor(
                cp.bipolar_electrodialysis_costing.factor_membrane_replacement,
                1,
            )
            iscale.set_scaling_factor(
                cp.bipolar_electrodialysis_costing.factor_stack_electrode_replacement,
                1,
            )
            for _, bpmed in self.bpmed.items():
                iscale.set_scaling_factor(
                    bpmed.costing.rectifier_cost_coeff[0], 1 / 100
                )
                iscale.set_scaling_factor(bpmed.costing.ac_power, work_scale * 1000)
                iscale.set_scaling_factor(
                    bpmed.costing.rectifier_cost_coeff[1], 1 / 1000
                )
                iscale.constraint_scaling_transform(
                    bpmed.costing.power_conversion, work_scale * 1000
                )
                scu.calculate_scale_from_dependent_vars(
                    bpmed.costing.capital_cost_rectifier,
                    bpmed.costing.capital_cost_rectifier_constraint,
                    [
                        bpmed.costing.rectifier_cost_coeff[0],
                        bpmed.costing.rectifier_cost_coeff[1],
                        bpmed.costing.ac_power,
                    ],
                )

                scu.calculate_scale_from_dependent_vars(
                    bpmed.costing.capital_cost,
                    bpmed.costing.capital_cost_constraint,
                    [
                        cp.bipolar_electrodialysis_costing.stack_electrode_capital_cost,
                        cp.bipolar_electrodialysis_costing.membrane_capital_cost,
                        bpmed.costing.capital_cost_rectifier,
                        bpmed.cell_triplet_num,
                        bpmed.cell_width,
                        bpmed.cell_length,
                    ],
                )
                scu.calculate_scale_from_dependent_vars(
                    bpmed.costing.fixed_operating_cost,
                    bpmed.costing.fixed_operating_cost_constraint,
                    [
                        cp.bipolar_electrodialysis_costing.stack_electrode_capital_cost,
                        cp.bipolar_electrodialysis_costing.membrane_capital_cost,
                        cp.bipolar_electrodialysis_costing.factor_stack_electrode_replacement,
                        cp.bipolar_electrodialysis_costing.factor_membrane_replacement,
                        bpmed.cell_triplet_num,
                        bpmed.cell_width,
                        bpmed.cell_length,
                    ],
                )
            for pump in [self.acidate_pump, self.basate_pump, self.brine_pump]:
                scu.calculate_scale_from_dependent_vars(
                    pump.costing.capital_cost,
                    pump.costing.capital_cost_constraint,
                    [pump.control_volume.work[0]],
                )
            if self.config.add_feed_bleed_for_acid_base:
                for pump in [
                    self.acidate_recycle_pump,
                    self.basate_recycle_pump,
                ]:
                    scu.calculate_scale_from_dependent_vars(
                        pump.costing.capital_cost,
                        pump.costing.capital_cost_constraint,
                        [pump.control_volume.work[0]],
                    )
            if self.config.add_feed_bleed_for_brine:
                for pump in [
                    self.diluate_recycle_pump,
                ]:
                    scu.calculate_scale_from_dependent_vars(
                        pump.costing.capital_cost,
                        pump.costing.capital_cost_constraint,
                        [pump.control_volume.work[0]],
                    )
            if self.config.add_mvc_concentrators and self.config.recycle_mvc_distillate:
                for pump in [self.acidate_distillate_pump, self.basate_distillate_pump]:
                    scu.calculate_scale_from_dependent_vars(
                        pump.costing.capital_cost,
                        pump.costing.capital_cost_constraint,
                        [pump.control_volume.work[0]],
                    )

    def init_translator_block(self, block, additonal_var_to_fix=None):
        """initializes translator block"""
        if additonal_var_to_fix is not None:
            if not isinstance(additonal_var_to_fix, list):
                additonal_var_to_fix = [additonal_var_to_fix]
            for var in additonal_var_to_fix:
                var.fix()
        flags = fix_state_vars(block.properties_in)
        solver = get_solver()
        results = solver.solve(block, tee=False)
        assert_optimal_termination(results)
        revert_state_vars(block.properties_in, flags)
        if additonal_var_to_fix is not None:
            for var in additonal_var_to_fix:
                var.unfix()

    def initialize_mixer(self, mixer, recycle_rate=1, **kwargs):
        fixed_streams = []
        stream_init = {}

        ref_stream = None
        if (
            self.config.add_feed_bleed_for_acid_base
            or self.config.add_feed_bleed_for_brine
        ):
            stream_init["recycle"] = True
        if self.config.add_mvc_concentrators:
            stream_init["mvr_distillate"] = True
        stream_init["feed"] = False

        inlet_var = mixer.find_component(f"{'feed'}_state")[0]
        ref_inlet = "feed"
        ref_stream = inlet_var
        for inlet, init_state in stream_init.items():
            if (
                init_state
                and ref_stream is not None
                and mixer.find_component(f"{inlet}_state") is not None
            ):
                inlet_var = mixer.find_component(f"{inlet}_state")[0]
                for idx, obj in inlet_var.flow_mol_phase_comp.items():
                    obj.fix(ref_stream.flow_mol_phase_comp[idx].value * recycle_rate)
                    fixed_streams.append(obj)
                inlet_var.pressure = ref_stream.pressure.value
                print("Propagated stream is ", ref_inlet)
                inlet_var.temperature.value = mixer.find_component(
                    f"{ref_inlet}_state"
                )[0].temperature.value
                print(
                    "Fixed temperature for inlet ",
                    inlet_var.temperature.value,
                )
        mixer.mixed_state[0].temperature.value = mixer.find_component(
            f"{ref_inlet}_state"
        )[0].temperature.value
        mixer.initialize()

        for stream in fixed_streams:
            stream.unfix()

    def initialize_recycle_pump(self, pump, inelet_pressure_guess):
        pump.control_volume.deltaP[0].fix(value(inelet_pressure_guess) - 101325)
        solver = get_solver()
        pump.inlet.fix()
        results = solver.solve(pump, tee=False)
        assert_optimal_termination(results)
        pump.inlet.unfix()
        # pump.initialize()
        pump.control_volume.deltaP[0].unfix()

    def initialize_unit(self, **kwargs):

        # Initialzie brine
        if self.config.add_feed_bleed_for_brine:
            self.brine_pump.deltaP[0].fix()
            self.brine_pump.initialize()
            propagate_state(self.brine_pump_to_brine_mixer)
            self.initialize_mixer(self.brine_mixer, recycle_rate=0.1)
            if self.default_property_package_diff_from_base:
                propagate_state(self.brine_mixer_to_brine_translator)
                self.init_translator_block(self.brine_translator_in)
                propagate_state(self.brine_translator_to_bpmed)
            else:
                propagate_state(self.brine_mixer_to_bpmed)

        else:
            self.brine_pump.deltaP[0].fix()
            self.brine_pump.initialize()
            if self.default_property_package_diff_from_base:
                propagate_state(self.brine_pump_to_brine_translator)
                self.init_translator_block(self.brine_translator_in)
                propagate_state(self.brine_translator_to_bpmed)
            else:
                propagate_state(self.brine_pump_to_bpmed)

        # initialize acid/base feed water
        if self.default_property_package_diff_from_base:
            self.init_translator_block(self.dilute_water_translator_in)
            propagate_state(self.dilute_water_to_splitter)
        self.low_tds_splitter.initialize()
        propagate_state(self.splitter_to_acidate_pump)
        propagate_state(self.splitter_to_basate_pump)
        self.acidate_pump.deltaP[0].fix()
        self.basate_pump.deltaP[0].fix()
        self.acidate_pump.initialize()
        self.basate_pump.initialize()
        if self.config.add_feed_bleed_for_acid_base or (
            self.config.add_mvc_concentrators and self.config.recycle_mvc_distillate
        ):
            propagate_state(self.acidate_pump_to_mixer)
            propagate_state(self.basate_pump_to_mixer)
            self.initialize_mixer(self.acidate_mixer, recycle_rate=0.1)
            self.initialize_mixer(self.basate_mixer, recycle_rate=0.1)
            propagate_state(self.basate_mixer_to_bpmed)
            propagate_state(self.acidate_mixer_to_bpmed)
        else:
            propagate_state(self.acidate_pump_to_bpmed)
            propagate_state(self.basate_pump_to_bpmed)

        self.acidate_pump.deltaP[0].unfix()
        self.basate_pump.deltaP[0].unfix()
        self.brine_pump.deltaP[0].unfix()

        # initalize bmped stages

        for idx in self.bpmed_stages:

            bpmed = self.bpmed[idx]
            bpmed.voltage_applied[0].unfix()
            bpmed.cell_length.unfix()
            bpmed.cell_width.unfix()
            bpmed.optob = Objective(
                expr=(bpmed.voltage_applied[0] - 150) ** 2
                + (bpmed.velocity_diluate[0, 0] - 0.05) ** 2
                + (
                    bpmed.outlet_diluate.flow_mol_phase_comp[0, "Liq", "Na_+"]
                    / bpmed.inlet_diluate.flow_mol_phase_comp[0, "Liq", "Na_+"]
                    - 0.5
                )
                ** 2
            )
            bpmed.outlet_diluate.pressure[0].unfix()
            bpmed.outlet_acidate.pressure[0].unfix()
            bpmed.outlet_basate.pressure[0].unfix()
            bpmed.initialize()
            bpmed.del_component("optob")
            if self.config.bpmed_stages > 1 and idx < self.bpmed_stages[-1]:
                propagate_state(self.find_component(f"sts_{idx}_diluate"))
                propagate_state(self.find_component(f"sts_{idx}_acidate"))
                propagate_state(self.find_component(f"sts_{idx}_basate"))
        self.bpmed[self.bpmed_stages[-1]].outlet_diluate.pressure[0].fix(101325)
        self.bpmed[self.bpmed_stages[-1]].outlet_acidate.pressure[0].fix(101325)
        self.bpmed[self.bpmed_stages[-1]].outlet_basate.pressure[0].fix(101325)
        # self.report()
        if self.config.add_feed_bleed_for_acid_base:
            propagate_state(self.bpmed_to_acidate_splitter)
            propagate_state(self.bpmed_to_basate_splitter)
            self.acidate_splitter.initialize()
            self.basate_splitter.initialize()
            propagate_state(self.acidate_splitter_to_recycle_pump)
            self.initialize_recycle_pump(
                self.acidate_recycle_pump, self.bpmed[0].inlet_acidate.pressure[0]
            )

            propagate_state(self.acidate_recycle_pump_to_mixer)
            propagate_state(self.basate_splitter_to_recycle_pump)
            self.initialize_recycle_pump(
                self.basate_recycle_pump, self.bpmed[0].inlet_basate.pressure[0]
            )
            propagate_state(self.basate_recycle_pump_to_mixer)
            if self.config.add_mvc_concentrators:
                propagate_state(self.acidate_splitter_to_acidate_mvc)
                propagate_state(self.basate_splitter_to_basate_mvc)
        elif self.config.add_mvc_concentrators:
            propagate_state(self.bpmed_to_acidate_mvc)
            propagate_state(self.bpmed_to_basate_mvc)
        if self.config.add_mvc_concentrators:
            self.acidate_mvc.initialize()
            self.basate_mvc.initialize()
            if self.config.recycle_mvc_distillate:
                self.initialize_recycle_pump(
                    self.acidate_distillate_pump,
                    self.bpmed[0].inlet_acidate.pressure[0],
                )
                self.initialize_recycle_pump(
                    self.basate_distillate_pump, self.bpmed[0].inlet_basate.pressure[0]
                )
                propagate_state(self.acidate_distillate_pump_to_acidate_mixer)
                propagate_state(self.basate_distillate_pump_to_basate_mixer)
            else:
                self.distillate_mixer.initialize()
        if self.default_property_package_diff_from_base:
            propagate_state(self.bpmed_diluate_to_translator_out)
            self.init_translator_block(self.diluate_translator_out)

        if self.config.add_feed_bleed_for_brine:
            if self.default_property_package_diff_from_base:
                propagate_state(self.diluate_translator_to_splitter)
            self.diluate_splitter.initialize()
            propagate_state(self.diluate_splitter_to_recycle_pump)
            self.initialize_recycle_pump(
                self.diluate_recycle_pump, self.bpmed[0].inlet_diluate.pressure[0]
            )
            propagate_state(self.diluate_recycle_pump_to_mixer)

        # if self.config.add_feed_bleed or self.config.add_mvc_concentrators:
        if self.config.add_mvc_concentrators:
            self.acidate_mvc.set_optimization_operation()
            self.basate_mvc.set_optimization_operation()
            self.optob = Objective(
                expr=sum(
                    (self.bpmed[s].voltage_applied[0] - 250) ** 2
                    for s in self.bpmed_stages
                )
                + sum(
                    (self.bpmed[s].velocity_diluate[0, 0] - 0.05) ** 2
                    for s in self.bpmed_stages
                )
                + ((self.nacl_recovery - 0.5) * 10) ** 2
                + ((self.basate_mvc.recovery - 0.75) * 10) ** 2
                + ((self.acidate_mvc.recovery - 0.75) * 10) ** 2
                + (self.basate_mvc.evaporator.area) ** 2
                + (self.basate_mvc.hx_distillate.area) ** 2
                + (self.basate_mvc.hx_brine.area) ** 2
                + (self.basate_mvc.compressor.pressure_ratio) ** 2
                + (self.acidate_mvc.evaporator.area) ** 2
                + (self.acidate_mvc.hx_distillate.area) ** 2
                + (self.acidate_mvc.hx_brine.area) ** 2
                + (self.acidate_mvc.compressor.pressure_ratio) ** 2
            )
            # self.activate_product_quality_constraints()
            self.acidate_mvc.recovery.setub(0.95)
            self.basate_mvc.recovery.setub(0.95)
            self.acidate_mvc.recovery.setlb(0.5)
            self.basate_mvc.recovery.setlb(0.5)
        else:
            self.optob = Objective(
                expr=sum(
                    (self.bpmed[s].voltage_applied[0] - 150) ** 2
                    for s in self.bpmed_stages
                )
                + sum(
                    (self.bpmed[s].velocity_diluate[0, 0] - 0.05) ** 2
                    for s in self.bpmed_stages
                )
                + ((self.nacl_recovery - 0.5) * 10) ** 2
            )
        for s in self.bpmed_stages:
            self.bpmed[s].cell_triplet_num.unfix()
        self.low_tds_water_inlet.fix()
        self.brine_inlet.fix()
        self.activate_equal_product_mass_flow_constraint()
        calculate_variable_from_constraint(self.nacl_recovery, self.eq_nacl_recovery)
        solver = get_solver()
        result = solver.solve(self, tee=True)
        self.report()
        assert_optimal_termination(result)
        if self.config.add_mvc_concentrators:
            self.acidate_mvc.set_fixed_operation()
            self.basate_mvc.set_fixed_operation()
            self.deactivate_product_quality_constraints()
        self.del_component("optob")
        self.low_tds_water_inlet.unfix()
        self.brine_inlet.unfix()
        for s in self.bpmed_stages:
            self.bpmed[s].voltage_applied[0].fix()
            self.bpmed[s].cell_length.fix()
            self.bpmed[s].cell_width.fix()
            self.bpmed[s].cell_triplet_num.fix()
        print("BP-MED model initialized successfully.")

    def get_model_state_dict(self):
        def get_ion_comp(stream, pH=None, pE=None):
            data_dict = dict()
            data_dict["Mass flow of H2O"] = stream.flow_mass_phase_comp["Liq", "H2O"]
            for phase, ion in stream.conc_mass_phase_comp:
                if ion != "H2O":
                    data_dict[ion] = stream.conc_mass_phase_comp[phase, ion]
            if pH is not None:
                data_dict["pH"] = pH
            if pE is not None:
                data_dict["pE"] = pE
            data_dict["Temperature"] = stream.temperature
            data_dict["Pressure"] = stream.pressure
            return data_dict

        def get_flow_mol_comp(stream):
            data_dict = dict()
            for phase, ion in stream.flow_mol_phase_comp:
                data_dict[ion] = stream.flow_mol_phase_comp[phase, ion]
            data_dict["Temperature"] = stream.temperature
            data_dict["Pressure"] = stream.pressure
            return data_dict

        def get_pe(port):
            if self.config.track_pE:
                return self.crystallizer.pE[port]
            else:
                return None

        model_state_dict = {
            "DI splitter": self.low_tds_splitter.split_fraction,
            "Pump pressure": {
                "Brine": pyunits.convert(
                    self.brine_pump.outlet.pressure[0], pyunits.bar
                ),
                "Acidate": pyunits.convert(
                    self.acidate_pump.outlet.pressure[0], pyunits.bar
                ),
                "Basate": pyunits.convert(
                    self.basate_pump.outlet.pressure[0], pyunits.bar
                ),
            },
        }
        for stage in self.bpmed_stages:
            model_state_dict.update(
                {
                    f"s{stage} EBDM brine inlet": get_flow_mol_comp(
                        self.bpmed[stage].diluate.properties[0, 0]
                    ),
                    f"s{stage} EBDM brine outlet": get_flow_mol_comp(
                        self.bpmed[stage].diluate.properties[0, 1]
                    ),
                    f"s{stage} EBDM acidate inlet": get_flow_mol_comp(
                        self.bpmed[stage].acidate.properties[0, 0]
                    ),
                    f"s{stage} EBDM acidate outlet": get_flow_mol_comp(
                        self.bpmed[stage].acidate.properties[0, 1]
                    ),
                    f"s{stage} EBDM basate inlet": get_flow_mol_comp(
                        self.bpmed[stage].basate.properties[0, 0]
                    ),
                    f"s{stage} EBDM basate outlet": get_flow_mol_comp(
                        self.bpmed[stage].basate.properties[0, 1]
                    ),
                    f"s{stage} EBDM": {
                        "Cell brine channel height": self.bpmed[stage].channel_height[
                            "diluate"
                        ],
                        "Cell acid/base channel height": self.bpmed[
                            stage
                        ].channel_height["acidate"],
                        "Cell width": self.bpmed[stage].cell_width,
                        "Cell length": self.bpmed[stage].cell_length,
                        "Cell triplet num": self.bpmed[stage].cell_triplet_num,
                        "Electrical stage num": self.bpmed[stage].electrical_stage_num,
                        "Brine inlet velocity": self.bpmed[stage].velocity_diluate[
                            0, 0
                        ],
                        "Basate inlet velocity": self.bpmed[stage].velocity_basate[
                            0, 0
                        ],
                        "Acidate inlet velocity": self.bpmed[stage].velocity_acidate[
                            0, 0
                        ],
                        "Current": self.bpmed[stage].current_density_x[0, 0],
                        "Voltage": self.bpmed[stage].voltage_applied[0],
                        "Limiting current density": self.bpmed[
                            stage
                        ].current_dens_lim_bpem[0, 1],
                    },
                }
            )
        model_state_dict.update(
            {
                "Product": self.flow_mass_product,
                "Mass fracitons of product": self.product_mass_concentration,
                "Overall EBDM NaCl recovery": self.nacl_recovery,
            }
        )
        if (
            self.config.add_feed_bleed_for_acid_base
            or self.config.add_feed_bleed_for_brine
            or (
                self.config.add_mvc_concentrators and self.config.recycle_mvc_distillate
            )
        ):
            mixer_inlets = ["feed"]
            if self.config.add_feed_bleed_for_acid_base:
                mixers = [self.acidate_mixer, self.basate_mixer]
                mixer_names = ["Acidate mixer", "Basate mixer"]
            if (
                self.config.add_feed_bleed_for_acid_base
                or self.config.add_feed_bleed_for_brine
            ):
                mixer_inlets.append("recycle")
            if self.config.add_feed_bleed_for_brine:
                mixers.append(self.brine_mixer)
                mixer_names.append("Brine mixer")
            if self.config.add_mvc_concentrators and self.config.recycle_mvc_distillate:
                mixer_inlets.append("mvr_distillate")

            for mixer, mixer_name in zip(
                mixers,
                mixer_names,
            ):
                for inlet in mixer_inlets:
                    model_state_dict[
                        mixer_name + " " + inlet.replace("_", " ").capitalize()
                    ] = get_ion_comp(
                        mixer.find_component(f"{inlet}_state")[0],
                    )
        if self.config.add_feed_bleed_for_acid_base:
            model_state_dict["Acidate recycle"] = self.acidate_splitter.split_fraction
            model_state_dict["Basate recycle"] = self.basate_splitter.split_fraction
            self.acidate_splitter.bleed_state[0].flow_mol_phase_comp.display()
            self.basate_splitter.bleed_state[0].flow_mol_phase_comp.display()
        if self.config.add_feed_bleed_for_brine:
            model_state_dict["Brine recycle"] = self.diluate_splitter.split_fraction
        if self.config.add_mvc_concentrators:
            self.acidate_mvc.report()
            self.basate_mvc.report()
        return model_state_dict
