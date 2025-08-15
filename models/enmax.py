from typing import TypeVar
from pydantic import BaseModel, Field, computed_field
from models.core.types import Input, Calculation, Output
from models.mixin import CommonModelMixin

NdArray = TypeVar("numpy.ndarray")


class D100(CommonModelMixin, BaseModel):
    """
    ENMAX D100 (Residential): service + energy + distribution + transmission.
    """
    consumption: Input                   # kWh total
    days: Input                          # billing days
    service_charge: Input = Field(..., description="$CAD/day")
    energy_charge: Input = Field(..., description="$CAD/kWh")
    distribution_variable_charge: Input = Field(..., description="$CAD/kWh")
    transmission_variable_charge: Input = Field(..., description="$CAD/kWh")

    @computed_field
    @property
    def consumption_charge(self) -> Calculation:
        arr: NdArray = self.consumption * (
            self.energy_charge
          + self.distribution_variable_charge
          + self.transmission_variable_charge
        )
        return Calculation(
            scenario=self.scenario,
            label="Consumption Charge",
            description="(energy + distribution + transmission) × consumption",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def distribution_charge(self) -> Calculation:
        arr: NdArray = self.days * self.service_charge
        return Calculation(
            scenario=self.scenario,
            label="Service Charge",
            description="service charge × days",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def total_cost(self) -> Output:
        tot = self.consumption_charge.data + self.distribution_charge.data
        return Output(
            scenario=self.scenario,
            label="D100 Total Cost",
            description="Total cost under D100",
            units="$CAD",
            data=tot,
        )


class D200(CommonModelMixin, BaseModel):
    """
    Energy cost under ENMAX D200: consumption + service & facility charges.
    """

    # inputs
    consumption: Input  # kWh in billing period
    days: Input  # number of days in billing period
    service_facility_charge: Input = Field(
        ..., description="Service & facility charge ($/day)"
    )
    usage_charge: Input = Field(..., description="System usage charge ($/kWh)")
    transmission_charge: Input = Field(
        ..., description="Transmission variable charge ($/kWh)"
    )

    @computed_field
    @property
    def consumption_charge(self) -> Calculation:
        arr: NdArray = self.consumption * (self.usage_charge + self.transmission_charge)
        return Calculation(
            scenario=self.scenario,
            label="Consumption Charge",
            description="(usage + transmission) * consumption",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def distribution_charge(self) -> Calculation:
        arr: NdArray = self.days * self.service_facility_charge
        return Calculation(
            scenario=self.scenario,
            label="Distribution Charge",
            description="service & facility charge * days",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def total_cost(self) -> Output:
        total = self.consumption_charge.data + self.distribution_charge.data
        return Output(
            scenario=self.scenario,
            label="D200 Total Cost",
            description="Total cost under D200 schedule",
            units="$CAD",
            data=total,
        )


class D300(CommonModelMixin, BaseModel):
    """
    Energy cost under ENMAX D300: consumption + demand + service charges.
    """

    # inputs
    consumption: Input  # kWh in billing period
    billing_demand: Input  # kVA billing demand
    days: Input  # number of days in billing period
    service_charge: Input = Field(..., description="Flat service charge ($/day)")
    facilities_charge: Input = Field(
        ..., description="Facilities charge ($/day per kVA)"
    )
    nonratchet_demand_charge: Input = Field(
        ..., description="Non-ratchet demand charge ($/day per kVA)"
    )
    transmission_demand_charge: Input = Field(
        ..., description="Transmission demand charge ($/day per kVA)"
    )
    variable_charge: Input = Field(..., description="Variable charge ($/kWh)")

    @computed_field
    @property
    def consumption_charge(self) -> Calculation:
        arr: NdArray = self.consumption * self.variable_charge
        return Calculation(
            scenario=self.scenario,
            label="Consumption Charge",
            description="variable charge * consumption",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def demand_charge(self) -> Calculation:
        per_kva = (
            self.facilities_charge
            + self.nonratchet_demand_charge
            + self.transmission_demand_charge
        )
        arr: NdArray = self.billing_demand * per_kva * self.days
        return Calculation(
            scenario=self.scenario,
            label="Demand Charge",
            description="(facilities + non-ratchet + transmission) * demand * days",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def service_charge_total(self) -> Calculation:
        arr: NdArray = self.days * self.service_charge
        return Calculation(
            scenario=self.scenario,
            label="Service Charge",
            description="service charge * days",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def total_cost(self) -> Output:
        total = (
            self.consumption_charge.data
            + self.demand_charge.data
            + self.service_charge_total.data
        )
        return Output(
            scenario=self.scenario,
            label="D300 Total Cost",
            description="Total cost under D300 schedule",
            units="$CAD",
            data=total,
        )



class D310(CommonModelMixin, BaseModel):
    """
    ENMAX D310 (Medium Commercial TOU): TOU energy + demand + service.
    """
    consumption_offpeak: Input
    consumption_shoulder: Input
    consumption_peak: Input
    billing_demand: Input
    days: Input
    service_charge: Input = Field(..., description="$CAD/day")
    facilities_charge: Input = Field(..., description="$CAD/day·kVA")
    nonratchet_demand_charge: Input = Field(..., description="$CAD/day·kVA")
    transmission_demand_charge: Input = Field(..., description="$CAD/day·kVA")
    variable_offpeak_charge: Input = Field(..., description="$CAD/kWh")
    variable_shoulder_charge: Input = Field(..., description="$CAD/kWh")
    variable_peak_charge: Input = Field(..., description="$CAD/kWh")

    @computed_field
    @property
    def energy_charge(self) -> Calculation:
        arr: NdArray = (
            self.consumption_offpeak * self.variable_offpeak_charge
          + self.consumption_shoulder * self.variable_shoulder_charge
          + self.consumption_peak * self.variable_peak_charge
        )
        return Calculation(
            scenario=self.scenario,
            label="Energy Charge",
            description="TOU consumption × TOU rates",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def demand_charge(self) -> Calculation:
        per_kva = (
            self.facilities_charge
          + self.nonratchet_demand_charge
          + self.transmission_demand_charge
        )
        arr: NdArray = self.billing_demand * per_kva * self.days
        return Calculation(
            scenario=self.scenario,
            label="Demand Charge",
            description="(facilities+non‑ratchet+transmission) × demand × days",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def service_charge_total(self) -> Calculation:
        arr: NdArray = self.days * self.service_charge
        return Calculation(
            scenario=self.scenario,
            label="Service Charge",
            description="service charge × days",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def total_cost(self) -> Output:
        tot = (
            self.energy_charge.data
          + self.demand_charge.data
          + self.service_charge_total.data
        )
        return Output(
            scenario=self.scenario,
            label="D310 Total Cost",
            description="Total cost under D310",
            units="$CAD",
            data=tot,
        )


class D410(CommonModelMixin, BaseModel):
    """
    ENMAX D410 (Large Commercial – ratcheted demand): energy + ratchet‑based demand + service.
    """
    consumption: Input
    billing_demand: Input        # current period kVA
    ratchet_demand: Input        # highest kVA in past 12 months
    days: Input
    service_charge: Input = Field(..., description="$CAD/day")
    facilities_charge: Input = Field(..., description="$CAD/day·kVA")
    ratchet_demand_charge: Input = Field(..., description="$CAD/day·kVA")
    transmission_demand_charge: Input = Field(..., description="$CAD/day·kVA")
    variable_charge: Input = Field(..., description="$CAD/kWh")

    @computed_field
    @property
    def energy_charge(self) -> Calculation:
        arr: NdArray = self.consumption * self.variable_charge
        return Calculation(
            scenario=self.scenario,
            label="Energy Charge",
            description="variable × consumption",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def demand_charge(self) -> Calculation:
        # apply ratchet if specified by user’s choice of billing vs ratchet demand
        per_kva = self.facilities_charge + self.ratchet_demand_charge + self.transmission_demand_charge
        arr: NdArray = self.ratchet_demand * per_kva * self.days
        return Calculation(
            scenario=self.scenario,
            label="Demand Charge",
            description="(facilities+ratchet+transmission) × ratchet demand × days",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def service_charge_total(self) -> Calculation:
        arr: NdArray = self.days * self.service_charge
        return Calculation(
            scenario=self.scenario,
            label="Service Charge",
            description="service charge × days",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def total_cost(self) -> Output:
        tot = (
            self.energy_charge.data
          + self.demand_charge.data
          + self.service_charge_total.data
        )
        return Output(
            scenario=self.scenario,
            label="D410 Total Cost",
            description="Total cost under D410",
            units="$CAD",
            data=tot,
        )


class D500(CommonModelMixin, BaseModel):
    """
    ENMAX D500 (Large Industrial TOU – ratcheted demand): TOU energy + ratchet demand + service.
    """
    consumption_offpeak: Input
    consumption_shoulder: Input
    consumption_peak: Input
    billing_demand: Input
    ratchet_demand: Input
    days: Input
    service_charge: Input = Field(..., description="$CAD/day")
    facilities_charge: Input = Field(..., description="$CAD/day·kVA")
    ratchet_demand_charge: Input = Field(..., description="$CAD/day·kVA")
    transmission_demand_charge: Input = Field(..., description="$CAD/day·kVA")
    variable_offpeak_charge: Input = Field(..., description="$CAD/kWh")
    variable_shoulder_charge: Input = Field(..., description="$CAD/kWh")
    variable_peak_charge: Input = Field(..., description="$CAD/kWh")

    @computed_field
    @property
    def energy_charge(self) -> Calculation:
        arr: NdArray = (
            self.consumption_offpeak * self.variable_offpeak_charge
          + self.consumption_shoulder * self.variable_shoulder_charge
          + self.consumption_peak * self.variable_peak_charge
        )
        return Calculation(
            scenario=self.scenario,
            label="Energy Charge",
            description="TOU cons × TOU rates",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def demand_charge(self) -> Calculation:
        per_kva = self.facilities_charge + self.ratchet_demand_charge + self.transmission_demand_charge
        arr: NdArray = self.ratchet_demand * per_kva * self.days
        return Calculation(
            scenario=self.scenario,
            label="Demand Charge",
            description="(facilities+ratchet+transmission) × ratchet × days",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def service_charge_total(self) -> Calculation:
        arr: NdArray = self.days * self.service_charge
        return Calculation(
            scenario=self.scenario,
            label="Service Charge",
            description="service charge × days",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def total_cost(self) -> Output:
        tot = (
            self.energy_charge.data
          + self.demand_charge.data
          + self.service_charge_total.data
        )
        return Output(
            scenario=self.scenario,
            label="D500 Total Cost",
            description="Total cost under D500",
            units="$CAD",
            data=tot,
        )


class D600(CommonModelMixin, BaseModel):
    """
    ENMAX D600 (Street Lighting): flat service + energy.
    """
    consumption: Input
    days: Input
    service_charge: Input = Field(..., description="$CAD/day")
    energy_charge: Input = Field(..., description="$CAD/kWh")

    @computed_field
    @property
    def energy_cost(self) -> Calculation:
        arr: NdArray = self.consumption * self.energy_charge
        return Calculation(
            scenario=self.scenario,
            label="Energy Cost",
            description="energy charge × consumption",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def service_cost(self) -> Calculation:
        arr: NdArray = self.days * self.service_charge
        return Calculation(
            scenario=self.scenario,
            label="Service Cost",
            description="service charge × days",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def total_cost(self) -> Output:
        tot = self.energy_cost.data + self.service_cost.data
        return Output(
            scenario=self.scenario,
            label="D600 Total Cost",
            description="Total cost under D600",
            units="$CAD",
            data=tot,
        )


class D700(CommonModelMixin, BaseModel):
    """
    ENMAX D700 (Municipal Transit): demand + energy + service.
    """
    consumption: Input
    billing_demand: Input
    days: Input
    service_charge: Input = Field(..., description="$CAD/day")
    facilities_charge: Input = Field(..., description="$CAD/day·kVA")
    nonratchet_demand_charge: Input = Field(..., description="$CAD/day·kVA")
    transmission_demand_charge: Input = Field(..., description="$CAD/day·kVA")
    energy_charge: Input = Field(..., description="$CAD/kWh")

    @computed_field
    @property
    def energy_cost(self) -> Calculation:
        arr: NdArray = self.consumption * self.energy_charge
        return Calculation(
            scenario=self.scenario,
            label="Energy Cost",
            description="energy charge × consumption",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def demand_cost(self) -> Calculation:
        per_kva = self.facilities_charge + self.nonratchet_demand_charge + self.transmission_demand_charge
        arr: NdArray = self.billing_demand * per_kva * self.days
        return Calculation(
            scenario=self.scenario,
            label="Demand Cost",
            description="(facilities+non‑ratchet+transmission) × demand × days",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def service_cost(self) -> Calculation:
        arr: NdArray = self.days * self.service_charge
        return Calculation(
            scenario=self.scenario,
            label="Service Cost",
            description="service charge × days",
            units="$CAD",
            data=arr,
        )

    @computed_field
    @property
    def total_cost(self) -> Output:
        tot = (
            self.energy_cost.data
          + self.demand_cost.data
          + self.service_cost.data
        )
        return Output(
            scenario=self.scenario,
            label="D700 Total Cost",
            description="Total cost under D700",
            units="$CAD",
            data=tot,
        )
