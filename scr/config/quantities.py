from dataclasses import dataclass

from scr.utils.types_alias import Quantity, SunspotPart


@dataclass(frozen=True)
class QuantitySpec:
    name: str
    latex: str
    unit: str
    mean_col_template: str
    std_col_template: str

    @property
    def latex_mean(self) -> str:
        return rf"$\langle {self.latex[1:-1]} \rangle$"

    @property
    def latex_std(self) -> str:
        return rf"$\sigma_{{{self.latex[1:-1]}}}$"

    def mean_col(self, location_suffix: str) -> str:
        return self.mean_col_template.format(loc=location_suffix)

    def std_col(self, location_suffix: str) -> str:
        return self.std_col_template.format(loc=location_suffix)


@dataclass(frozen=True)
class LocationSpec:
    name: str
    suffix: str
    label: str


@dataclass(frozen=True)
class MeasurementSpec:
    quantity: QuantitySpec
    location: LocationSpec
    threshold: float = 0.0

    @property
    def mean_col(self) -> str:
        return self.quantity.mean_col(self.location.suffix)

    @property
    def std_col(self) -> str:
        return self.quantity.std_col(self.location.suffix)

    @property
    def ylabel_mean(self) -> str:
        if self.quantity.unit:
            return (
                f"{self.quantity.latex_mean} "
                f"({self.quantity.unit}, {self.location.label})"
            )
        return f"{self.quantity.latex_mean} ({self.location.label})"

    @property
    def ylabel_std(self) -> str:
        if self.quantity.unit:
            return (
                f"{self.quantity.latex_std} "
                f"({self.quantity.unit}, {self.location.label})"
            )
        return f"{self.quantity.latex_std} ({self.location.label})"


_QUANTITIES = {
    "Ic": QuantitySpec(
        name="Ic",
        latex=r"$I^\mathrm{c}/I^\mathrm{c}_\mathrm{QS}$",
        unit="",
        mean_col_template="Ic_{loc}_mean",
        std_col_template="Ic_{loc}_std",
    ),
    "B": QuantitySpec(
        name="B",
        latex=r"$B$",
        unit="G",
        mean_col_template="B_{loc}_mean",
        std_col_template="B_{loc}_std",
    ),
    "Bp": QuantitySpec(
        name="Bp",
        latex=r"$B_{\mathrm{p}}$",
        unit="G",
        mean_col_template="Bp_{loc}_mean",
        std_col_template="Bp_{loc}_std",
    ),
    "Bt": QuantitySpec(
        name="Bt",
        latex=r"$B_{\mathrm{t}}$",
        unit="G",
        mean_col_template="Bt_{loc}_mean",
        std_col_template="Bt_{loc}_std",
    ),
    "Bver": QuantitySpec(
        name="Bver",
        latex=r"$B_{\mathrm{ver}}$",
        unit="G",
        mean_col_template="Br_{loc}_mean",
        std_col_template="Br_{loc}_std",
    ),
    "Bhor": QuantitySpec(
        name="Bhor",
        latex=r"$B_{\mathrm{hor}}$",
        unit="G",
        mean_col_template="Bhor_{loc}_mean",
        std_col_template="Bhor_{loc}_std",
    ),
}

_QUANTITIES["Br"] = _QUANTITIES["Bver"]

_LOCATIONS = {
    "Ic<0.5": LocationSpec(
        name="umbra",
        suffix="Ic<0.5_corrected_border_flux",
        label="umbra boundary",
    ),
    "Ic<0.9-Ic<0.5": LocationSpec(
        name="penumbra",
        suffix="Ic<0.9-Ic<0.5_corrected_border_flux",
        label="penumbra outer boundary",
    ),
    "sunspot": LocationSpec(
        name="sunspot",
        suffix="sunspot_corrected_border_flux",
        label="sunspot boundary",
    ),
    "Ic<0.65": LocationSpec(
        name="pore",
        suffix="Ic<0.65_corrected_border_flux",
        label="pore boundary",
    ),
}

_DEFAULT_THRESHOLDS = {
    ("Ic", "Ic<0.9"): 0.9,
    ("Ic", "Ic<0.9-Ic<0.65"): 0.9,
    ("Ic", "Ic<0.9-Ic<0.5"): 0.9,
    ("Ic", "Ic<0.65"): 0.65,
    ("Ic", "Ic<0.65-Ic<0.5"): 0.65,
    ("Ic", "Ic<0.5"): 0.5,
    ("B", "Ic<0.9-Ic<0.5"): 605.1246290384582,
    ("Bhor", "Ic<0.9-Ic<0.5"): 599.659971407691,
}


def get_measurement_spec(
    quantity: Quantity,
    location: SunspotPart,
    threshold: float | None = None,
) -> MeasurementSpec:
    try:
        q = _QUANTITIES[quantity]
    except KeyError:
        raise ValueError(f"Unknown quantity: {quantity}")

    try:
        loc = _LOCATIONS[location]
    except KeyError:
        raise ValueError(f"Unknown location: {location}")

    if threshold is None:
        threshold = _DEFAULT_THRESHOLDS.get((quantity, location), 0.0)

    return MeasurementSpec(
        quantity=q,
        location=loc,
        threshold=threshold,
    )


def canonical_quantity_order() -> list[Quantity | str]:
    """
    Return canonical scientific order of quantities.
    Aliases (e.g. 'Br') are excluded.
    """
    # Exclude aliases by checking identity
    unique = []
    seen_ids = set()

    for name, spec in _QUANTITIES.items():
        if id(spec) in seen_ids:
            continue
        seen_ids.add(id(spec))
        unique.append(name)

    return unique


def order_quantities(
        quantities: list[Quantity | str]
) -> list[Quantity | str]:
    """
    Order quantities according to canonical configuration order.
    Unknown quantities are appended at the end.
    """
    canonical = canonical_quantity_order()
    present = set(quantities)

    ordered = [q for q in canonical if q in present]
    ordered += [q for q in quantities if q not in canonical]

    return ordered
