from typing_extensions import Final

from alfspy.core.util.defs import StrEnum


class Side(StrEnum):
    """
    Enum describing the side of an object.
    """
    Top: Final[str] = 'top'
    Right: Final[str] = 'right'
    Bottom: Final[str] = 'bottom'
    Left: Final[str] = 'left'
    Near: Final[str] = 'near'
    Far: Final[str] = 'far'