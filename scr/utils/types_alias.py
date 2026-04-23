from typing import TypeAlias, Literal
import numpy as np
from numpy.typing import NDArray
from astropy.io import fits

# ---------------------------------------------------------------------
# Low-level data containers
# ---------------------------------------------------------------------

Header: TypeAlias = fits.Header
Headers: TypeAlias = list[Header]

# Boolean or float masks (assumed 2D by convention)
Mask: TypeAlias = NDArray[np.bool_ | np.floating]
Masks: TypeAlias = list[Mask]

# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

# Point: TypeAlias = tuple[float, float]       # (x, y) or (col, row)
Contour: TypeAlias = NDArray[np.floating]      # One closed/open curve with shape (N, 2)
Contours: TypeAlias = list[Contour]            # Multiple contours per frame

# ---------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------

FrameID: TypeAlias = int
TrackID: TypeAlias = int
SunspotID: TypeAlias = int

ObservationID: TypeAlias = str

Quantity: TypeAlias = Literal["Ic", "B", "Bp", "Bt", "Br", "Bver", "Bhor"]
Metadata: TypeAlias = dict

SunspotPart: TypeAlias = str       # "umbra", "outer", ...
SunspotPhase: TypeAlias = Literal["forming", "stable", "decaying"]

Events: TypeAlias = list[dict]

Stat_key: TypeAlias = str
Stat_value: TypeAlias = float | int | list[float | int] | NDArray[np.floating]

# ---------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------

AssociationMode: TypeAlias = Literal["strict", "covers"]
ObjectFilteringMode: TypeAlias = Literal["sunspots", "pores", "all_sunspots", "all_pores"]
FrameFilteringMode: TypeAlias = Literal["frame-wise", "any", "all"]

# ---------------------------------------------------------------------
# Tracking structures (time evolution)
# ---------------------------------------------------------------------

Track: TypeAlias = dict[FrameID, Contours]
Tracks: TypeAlias = dict[TrackID, Track]
TracksLabeled: TypeAlias = dict[SunspotPart, Tracks]

# ---------------------------------------------------------------------
# Sunspot hierarchy
# ---------------------------------------------------------------------

Sunspot = dict[SunspotPart, Track]
Sunspots = dict[SunspotID, Sunspot]

SunspotPhases = dict[SunspotPhase, Sunspot]
SunspotsPhases = dict[SunspotID, SunspotPhases]

# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

Stat: TypeAlias = dict[Stat_key, Stat_value]

Stats: TypeAlias = dict[
    SunspotID,
    dict[SunspotPart, dict[FrameID, Stat]]
]

StatsByQuantity: TypeAlias = dict[Quantity, Stats]

# ---------------------------------------------------------------------
# Global collections
# ---------------------------------------------------------------------

SunspotsPhasesByObservation = dict[ObservationID, SunspotsPhases]
StatsByObservation = dict[ObservationID, StatsByQuantity]
