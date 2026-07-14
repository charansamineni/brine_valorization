from watertap.costing import WaterTAPCosting

import pyomo.environ as pyo


class ValorizationCostingBlock(WaterTAPCosting):
    def add_LCOP(self, mass_product, name="LCOP"):
        denominator = (
            pyo.units.convert(mass_product, to_units=pyo.units.kg / self.base_period)
            * self.utilization_factor
        )

        self.add_component(
            name,
            pyo.Expression(
                expr=(
                    self.total_capital_cost * self.capital_recovery_factor
                    + self.total_operating_cost
                )
                / denominator,
                doc=f"Levelized Cost of Product",
            ),
        )

    def add_mass_based_specific_energy_consumption(
        self, flow_rate, name="specific_energy_consumption"
    ):
        """
        Add specific energy consumption (kWh/kg) to costing block.
        Args:
            flow_rate - flow rate of water (mass-based) to be used in
                        calculating specific energy consumption
            name (optional) - the name of the Expression for the specific
                              energy consumption (default: specific_energy_consumption)
        """

        self.add_component(
            name,
            pyo.Expression(
                expr=self.aggregate_flow_electricity
                / pyo.units.convert(flow_rate, to_units=pyo.units.kg / pyo.units.hr),
                doc=f"Specific energy consumption based on product mass",
            ),
        )

    def add_annual_product_generation(
        self, flow_rate, name="annual_product_generation"
    ):
        """
        Add annual water production to costing block.
        Args:
            flow_rate - flow rate of water (volumetric) to be used in
                        calculating annual product generation
            name (optional) - name for the annual product generation variable
                              Expression (default: annual_product_generation)
        """
        self.add_component(
            name,
            pyo.Expression(
                expr=(
                    pyo.units.convert(
                        flow_rate, to_units=pyo.units.kg / self.base_period
                    )
                    * self.utilization_factor
                ),
                doc=f"Annual product generation based on flow ",
            ),
        )

    def register_product(self, name, cost_var):
        self.register_flow_type(name, -1 * cost_var)

    def register_product_flow(self, name, flow_var):
        self.cost_flow(flow_var, name)
