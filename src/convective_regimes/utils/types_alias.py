from typing import TypeAlias, Literal

FilterMode: TypeAlias = Literal["sunspots", "pores"]
SunspotPhase: TypeAlias = Literal["all", "forming", "stable", "decaying"]
Quantity: TypeAlias = Literal[
    "Ic", "B", "Bhor", "Binc", "Br",
    "Br_sunspot_flux_corr_total", "sunspot_area_corr",
    "spot_global_index", "frame"]
Boundary: TypeAlias = Literal["b605.0", "b550.0", "ic0.9", "ic0.5", "ic0.65"]
