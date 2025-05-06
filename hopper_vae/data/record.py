"""
    Organize sample collection and transmission light microscopy data
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional, Set, Tuple

from numpy import uint8


class WingSide(Enum):
    """
    Enum for wing side.
    """

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class WingType(Enum):
    """
    Enum for wing type.
    """

    FOREWING = "forewing"
    HINDWING = "hindwing"
    UNKNOWN = "unknown"


class PreserveMedium(Enum):
    """
    Enum for preservation medium.
    """

    ETHANOL = "ethanol"
    UNKNOWN = "unknown"


class ContainerType(Enum):
    """
    Enum for container type.
    """

    TUBE_15ML = "15ml tube"
    TUBE_50ML = "50ml tube"
    UNKNOWN = "unknown"


# TODO: add imaging instrument types and settings


@dataclass
class Metadata:
    """
    Dataclass to organize sample collection and transmission light microscopy data.
    """

    record_id: int = None
    specimen_id: str = None
    species: str = None
    year: int = None
    date_collected: str = None
    loc_tag: str = None
    coordinates: str = None
    collected_by: Optional[str] = None
    live_sample: bool = None
    imaged_by: Optional[str] = None
    sex: str = None
    preserve_medium: PreserveMedium = None
    dried: str = None
    container_type: Set[ContainerType] = None
    imaging_instrument: str = None
    date_imaged: str = None
    mag: str = None
    obj: str = None
    damaged: bool = None
    image_success: bool = None
    discarded: bool = None
    notes: str = None
    wing_type: WingType = None
    wing_side: WingSide = None
    replicate_number: uint8 = None
    best_in_set: bool = None
    image_size: Tuple[int, int] = None
    image_channels: tuple = None
    image_bit_depth: int = None
    image_type: str = None
    file_hash: str = None

    def to_dict(self) -> dict:
        """
        Convert the dataclass to a dictionary and map enum instances to their underlying values.
        """

        def convert(value: Any) -> Any:
            # If the value is an Enum, return its value.
            if isinstance(value, Enum):
                return value.value
            # If the value is a set, convert each element.
            elif isinstance(value, set):
                return {convert(item) for item in value}
            # If it's a list, convert each element.
            elif isinstance(value, list):
                return [convert(item) for item in value]
            # If it's a dict, convert its values.
            elif isinstance(value, dict):
                return {key: convert(val) for key, val in value.items()}
            else:
                return value

        raw_dict = asdict(self)
        return convert(raw_dict)
