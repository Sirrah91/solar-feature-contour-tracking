from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------------------
# Region geometry constants (belong here  they define the regions below)
# ---------------------------------------------------------------------------

SUNSPOT_PQ_B_THRESHOLD = 605.0

TRANSITION_BHOR_MIN = 900.0
TRANSITION_GAMMA_MIN = 40.0
TRANSITION_GAMMA_MAX = 55.0

LIGHT_BRIDGE_BHOR_MAX = 900.0
LIGHT_BRIDGE_GAMMA_MAX = 40.0
LIGHT_BRIDGE_BR_MIN = 800.0
LIGHT_BRIDGE_BR_MAX = 1200.0


# ---------------------------------------------------------------------------
# MagneticRegion
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MagneticRegion:
    """Definition of a magnetic regime region."""

    name: str

    bhor_min: float | None = None
    bhor_max: float | None = None

    binc_min: float | None = None
    binc_max: float | None = None

    br_min: float | None = None
    br_max: float | None = None

    ic_min: float | None = None
    ic_max: float | None = None

    def _build_masks(
            self,
            *,
            Bhor: np.ndarray | None = None,
            Binc: np.ndarray | None = None,
            Br: np.ndarray | None = None,
            Ic: np.ndarray | None = None,
            extra_mask: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        masks: list[np.ndarray] = []

        if Bhor is not None:
            if self.bhor_min is not None:
                masks.append(Bhor >= self.bhor_min)
            if self.bhor_max is not None:
                masks.append(Bhor <= self.bhor_max)

        if Binc is not None:
            if self.binc_min is not None:
                masks.append(Binc >= self.binc_min)
            if self.binc_max is not None:
                masks.append(Binc <= self.binc_max)

        if Br is not None:
            if self.br_min is not None:
                masks.append(Br >= self.br_min)
            if self.br_max is not None:
                masks.append(Br <= self.br_max)

        if Ic is not None:
            if self.ic_min is not None:
                masks.append(Ic >= self.ic_min)
            if self.ic_max is not None:
                masks.append(Ic <= self.ic_max)

        if extra_mask is not None:
            masks.append(extra_mask)

        if not masks:
            raise ValueError(
                f"No constraints were evaluated for region '{self.name}'. "
                "Pass at least one array matching a defined bound."
            )

        return masks

    def interior(
            self,
            *,
            Bhor: np.ndarray | None = None,
            Binc: np.ndarray | None = None,
            Br: np.ndarray | None = None,
            Ic: np.ndarray | None = None,
            extra_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Boolean mask of pixels inside this region.

        Only fields for which an array is supplied are evaluated.
        At least one array must be provided.
        """
        return np.logical_and.reduce(
            self._build_masks(Bhor=Bhor, Binc=Binc, Br=Br, Ic=Ic, extra_mask=extra_mask)
        )

    def exterior(
            self,
            *,
            Bhor: np.ndarray | None = None,
            Binc: np.ndarray | None = None,
            Br: np.ndarray | None = None,
            Ic: np.ndarray | None = None,
            extra_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Boolean mask of pixels outside this region.

        Each supplied quantity is evaluated against its bounds independently,
        then combined: a pixel is exterior if it fails any one constraint.
        This is the element-wise negation of interior.
        """
        return ~self.interior(
            Bhor=Bhor, Binc=Binc, Br=Br, Ic=Ic, extra_mask=extra_mask
        )

# ---------------------------------------------------------------------------
# Canonical region instances
# ---------------------------------------------------------------------------


TRANSITION_REGION = MagneticRegion(
    name="transition",
    bhor_min=TRANSITION_BHOR_MIN,
    binc_min=TRANSITION_GAMMA_MIN,
    binc_max=TRANSITION_GAMMA_MAX,
)

LIGHT_BRIDGE_REGION = MagneticRegion(
    name="light_bridge",
    bhor_max=LIGHT_BRIDGE_BHOR_MAX,
    binc_max=LIGHT_BRIDGE_GAMMA_MAX,
    br_min=LIGHT_BRIDGE_BR_MIN,
    br_max=LIGHT_BRIDGE_BR_MAX,
)

PORE_GAP_REGION = MagneticRegion(
    name="pore_gap",
    bhor_min=SUNSPOT_PQ_B_THRESHOLD,
    binc_min=55.0,
)

PENUMBRA = MagneticRegion(
    name="penumbra",
    ic_min=0.5,
    ic_max=0.9,
)
