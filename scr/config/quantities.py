from dataclasses import dataclass

from scr.utils.types_alias import Quantity, SunspotPart


@dataclass(frozen=True)
class QuantitySpec:
    name: str
    latex: str
    unit: str
    mean_col_template: str | None
    std_col_template: str | None

    @property
    def latex_mean(self) -> str:
        return rf"$\langle {self.latex[1:-1]} \rangle$"

    @property
    def latex_std(self) -> str:
        return rf"$\sigma_{{{self.latex[1:-1]}}}$"

    def mean_col(self, location_suffix: str) -> str:
        if self.mean_col_template is None:
            raise ValueError(
                f"No mean column template defined for quantity '{self.name}'."
            )
        return self.mean_col_template.format(loc=location_suffix)

    def std_col(self, location_suffix: str) -> str:
        if self.std_col_template is None:
            raise ValueError(
                f"No std column template defined for quantity '{self.name}'."
            )
        return self.std_col_template.format(loc=location_suffix)


@dataclass(frozen=True)
class LocationSpec:
    name: str
    suffix: str
    label: str


@dataclass(frozen=True)
class MeasurementSpec:
    quantity: QuantitySpec
    location: LocationSpec | None = None
    threshold: float | None = None

    # ── private helpers ───────────────────────────────────────────────

    def _dep(self, depends_on: list[QuantitySpec] | None) -> str:
        if depends_on is None:
            return ""
        inner = ", ".join(q.latex[1:-1] for q in depends_on)
        return rf"\left({inner}\right)"

    def _suffix(self) -> str:
        """Parenthesised unit + location clause, or ''."""
        parts = []
        if self.quantity.unit:
            parts.append(rf"\mathrm{{{self.quantity.unit}}}")
        if self.location:
            parts.append(rf"\mathrm{{{self.location.label}}}")
        return rf"\,\left({', '.join(parts)}\right)" if parts else ""

    def _make_label(
            self,
            inner: str,
            *,
            superscript: str | None = None,
            subscript: str | None = None,
            depends_on: list[QuantitySpec] | None = None,
    ) -> str:
        if superscript is not None:
            inner = rf"{inner}^{{\mathrm{{{superscript}}}}}"
        if subscript is not None:
            inner = rf"{inner}_{{\mathrm{{{subscript}}}}}"
        return f"${inner}{self._dep(depends_on)}{self._suffix()}$"

    # ── public labels ─────────────────────────────────────────────────

    @property
    def mean_col(self) -> str:
        if self.location is None:
            raise ValueError(
                f"mean_col is not defined for quantity-only spec '{self.quantity.name}'. "
                "Provide a location."
            )
        return self.quantity.mean_col(self.location.suffix)

    @property
    def std_col(self) -> str:
        if self.location is None:
            raise ValueError(
                f"std_col is not defined for quantity-only spec '{self.quantity.name}'. "
                "Provide a location."
            )
        return self.quantity.std_col(self.location.suffix)

    def label(
            self,
            *,
            superscript: str | None = None,
            subscript: str | None = None,
            depends_on: list[QuantitySpec] | None = None,
    ) -> str:
        return self._make_label(self.quantity.latex[1:-1],
                                superscript=superscript, subscript=subscript, depends_on=depends_on)

    def label_mean(
            self,
            *,
            superscript: str | None = None,
            subscript: str | None = None,
            depends_on: list[QuantitySpec] | None = None,
    ) -> str:
        return self._make_label(self.quantity.latex_mean[1:-1],
                                superscript=superscript, subscript=subscript, depends_on=depends_on)

    def label_std(
            self,
            *,
            superscript: str | None = None,
            subscript: str | None = None,
            depends_on: list[QuantitySpec] | None = None,
    ) -> str:
        return self._make_label(self.quantity.latex_std[1:-1],
                                superscript=superscript, subscript=subscript, depends_on=depends_on)


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
    "Binc": QuantitySpec(
        name="Binc",
        latex=r"$\gamma$",
        unit="deg",
        mean_col_template=None,
        std_col_template=None,
    ),
    "Phi": QuantitySpec(
        name="Flux",
        latex=r"$\Phi$",
        unit="Mx",
        mean_col_template=None,
        std_col_template=None,
    ),
}

_QUANTITIES["Br"] = _QUANTITIES["Bver"]

_LOCATIONS = {
    "Ic<0.9": LocationSpec(
        name="penumbra",
        suffix="Ic<0.9_flux-border_corr",
        label="penumbra outer boundary",
    ),
    "Ic<0.65": LocationSpec(
        name="pore",
        suffix="Ic<0.65_flux-border_corr",
        label="pore boundary",
    ),
    "Ic<0.5": LocationSpec(
        name="umbra",
        suffix="Ic<0.5_flux-border_corr",
        label="umbra boundary",
    ),
    "sunspot": LocationSpec(
        name="sunspot",
        suffix="sunspot_flux-border_corr",
        label="sunspot boundary",
    ),
    "internal_voids": LocationSpec(
        name="quiet Sun",
        suffix="internal_voids_flux-border_corr",
        label="quiet Sun / granulation",
    ),
}

_LOCATIONS["Ic<0.9-Ic<0.65"] = _LOCATIONS["Ic<0.9-Ic<0.5"] = _LOCATIONS["Ic<0.9"]
_LOCATIONS["Ic<0.65-Ic<0.5"] = _LOCATIONS["Ic<0.65"]

# Named aliases
_LOCATIONS["umbra"] = _LOCATIONS["Ic<0.5"]
_LOCATIONS["penumbra"] = _LOCATIONS["Ic<0.9-Ic<0.5"]
_LOCATIONS["QS"] \
    = _LOCATIONS["quietSun"] \
    = _LOCATIONS["granulation"] \
    = _LOCATIONS["internal_voids"]

_DEFAULT_THRESHOLDS = {
    ("Ic", "Ic<0.9"): 0.9,
    ("Ic", "Ic<0.65"): 0.65,
    ("Ic", "Ic<0.5"): 0.5,
    ("B", "Ic<0.9"): 605.1246290384582,
    ("Bhor", "Ic<0.9"): 599.659971407691,
}

# Ic<0.9 family
_DEFAULT_THRESHOLDS[("Ic", "Ic<0.9-Ic<0.65")] = \
    _DEFAULT_THRESHOLDS[("Ic", "Ic<0.9-Ic<0.5")] = \
    _DEFAULT_THRESHOLDS[("Ic", "sunspot")] = \
    _DEFAULT_THRESHOLDS[("Ic", "penumbra")] = _DEFAULT_THRESHOLDS[("Ic", "Ic<0.9")]

# Ic<0.65 family
_DEFAULT_THRESHOLDS[("Ic", "Ic<0.65-Ic<0.5")] = _DEFAULT_THRESHOLDS[("Ic", "Ic<0.65")]

# Ic<0.5 / umbra
_DEFAULT_THRESHOLDS[("Ic", "umbra")] = _DEFAULT_THRESHOLDS[("Ic", "Ic<0.5")]

# B family
_DEFAULT_THRESHOLDS[("B", "Ic<0.9-Ic<0.65")] = \
    _DEFAULT_THRESHOLDS[("B", "Ic<0.9-Ic<0.5")] = \
    _DEFAULT_THRESHOLDS[("B", "sunspot")] = \
    _DEFAULT_THRESHOLDS[("B", "penumbra")] = _DEFAULT_THRESHOLDS[("B", "Ic<0.9")]

# Bhor family
_DEFAULT_THRESHOLDS[("Bhor", "Ic<0.9-Ic<0.65")] = \
    _DEFAULT_THRESHOLDS[("Bhor", "Ic<0.9-Ic<0.5")] = \
    _DEFAULT_THRESHOLDS[("Bhor", "sunspot")] = \
    _DEFAULT_THRESHOLDS[("Bhor", "penumbra")] = _DEFAULT_THRESHOLDS[("Bhor", "Ic<0.9")]


def get_measurement_spec(
        quantity: str,
        location: SunspotPart | None = None,
        threshold: float | None = None,
) -> MeasurementSpec:
    try:
        q = _QUANTITIES[quantity]
    except KeyError:
        raise ValueError(f"Unknown quantity: {quantity}")

    if location is None:
        return MeasurementSpec(quantity=q, threshold=threshold)

    try:
        loc = _LOCATIONS[location]
    except KeyError:
        raise ValueError(f"Unknown location: {location}")

    if threshold is None:
        threshold = _DEFAULT_THRESHOLDS.get((quantity, location))

    return MeasurementSpec(quantity=q, location=loc, threshold=threshold)


def canonical_quantity_order() -> list[Quantity | str]:
    """
    Return canonical scientific order of quantities.
    Aliases (e.g. 'Br') are excluded.
    """
    unique = []
    seen_ids = set()

    for name, spec in _QUANTITIES.items():
        if id(spec) in seen_ids:
            continue
        seen_ids.add(id(spec))
        unique.append(name)

    return unique


def order_quantities(quantities: list[Quantity | str]) -> list[Quantity | str]:
    """
    Order quantities according to canonical configuration order.
    Unknown quantities are appended at the end.
    """
    canonical = canonical_quantity_order()
    present = set(quantities)

    ordered = [q for q in canonical if q in present]
    ordered += [q for q in quantities if q not in canonical]

    return ordered
