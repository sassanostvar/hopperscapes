"""
    Organize sample collection and transmitted light microscopy data
"""

from dataclasses import asdict, dataclass, field
from enum import Enum, unique
from typing import Any, Optional, Set, Tuple, Dict
from datetime import date

UNKNOWN = "<unknown>"


@unique
class WingSide(Enum):
    """
    Enum for wing side.
    """

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = UNKNOWN


@unique
class WingType(Enum):
    """
    Enum for wing type.
    """

    FOREWING = "forewing"
    HINDWING = "hindwing"
    UNKNOWN = UNKNOWN


@unique
class PreserveMedium(Enum):
    """
    Enum for preservation medium.
    """

    ETHANOL_70 = "ethanol_70"
    ETHANOL_95 = "ethanol_95"
    UNKNOWN = UNKNOWN


@unique
class ContainerType(Enum):
    """
    Enum for container type.
    """

    GREINER_CONICAL_50ML = "Greiner_conical_50ml"
    TEENITOR_TOY_15ML = "Teenitor_toy_15ml"
    ACKERS_CONICAL_15ML = "Ackers_conical_15ml"
    SIMPURE_CONICAL_15ML = "SimPure_conical_15ml"
    FALCON_CONICAL_15ML = "Falcon_conical_15ml"
    UNKNOWN_CONICAL_15ML = "Unknown_conical_15ml"
    UNKNOWN = UNKNOWN


@unique
class ImagingInstrument(Enum):
    """
    Enum for imaging instrument.
    """

    AMSCOPE = "AMSCOPE-SM-1TSZ-V203"
    LEICA = "Leica"
    UNKNOWN = UNKNOWN


# fmt:off
@dataclass
class Metadata:
    """
    Dataclass to organize sample collection and transmitted light microscopy data.
    """

    record_id           : Optional[int]             = None
    specimen_id         : Optional[str]             = None
    species             : Optional[str]             = None
    year                : Optional[int]             = None
    date_collected      : Optional[date]            = None
    site_code           : Optional[str]             = None
    coordinates         : Optional[str]             = None
    collected_by        : Optional[str]             = None
    live_sample         : Optional[bool]            = None
    imaged_by           : Optional[str]             = None
    sex                 : Optional[str]             = None
    preserve_medium     : PreserveMedium            = PreserveMedium.UNKNOWN
    dried               : Optional[str]             = None
    container_type      : Set[ContainerType]        = field(default_factory=set)
    imaging_instrument  : ImagingInstrument         = ImagingInstrument.UNKNOWN
    date_imaged         : Optional[date]            = None
    mag                 : Optional[str]             = None
    obj                 : Optional[str]             = None
    damaged             : Optional[bool]            = None
    image_success       : Optional[bool]            = None
    discarded           : Optional[bool]            = None
    notes               : Optional[str]             = None
    wing_type           : WingType                  = WingType.UNKNOWN
    wing_side           : WingSide                  = WingSide.UNKNOWN
    replicate_number    : Optional[int]             = None
    best_in_set         : Optional[bool]            = None
    image_size          : Optional[Tuple[int, int]] = None
    image_channels      : Optional[Tuple[int, ...]] = None
    image_bit_depth     : Optional[int]             = None
    image_file_ext      : Optional[str]             = None
    color_space         : Optional[str]             = None
    color_profile       : Optional[str]             = None
    image_resolution    : Optional[Tuple[int, int]] = None
    file_hash           : Optional[str]             = None

    def __post_init__(self):
        if isinstance(self.preserve_medium, str):
            self.preserve_medium = PreserveMedium(self.preserve_medium)

        if isinstance(self.container_type, (list, tuple)):
            self.container_type = {ContainerType(x) for x in self.container_type}

        if isinstance(self.imaging_instrument, str):
            self.imaging_instrument = ImagingInstrument(self.imaging_instrument)

        for field_name, enum_cls in (("wing_side", WingSide), ("wing_type", WingType)):
            val = getattr(self, field_name)
            if isinstance(val, str):
                setattr(self, field_name, enum_cls(val))

        for field_name in ("date_collected", "date_imaged"):
            val = getattr(self, field_name)
            if isinstance(val, str):
                try:
                    setattr(self, field_name, date.fromisoformat(val))
                except ValueError:
                    raise ValueError(
                        f"Invalid date format for {field_name}: {val}. "
                        "Expected format is YYYY-MM-DD."
                    )

    # fmt:on
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

    def to_pandas(self):
        """
        Convert the dataclass to a pandas DataFrame.
        """
        import pandas as pd

        return pd.DataFrame([self.to_dict()])

    def to_json(self) -> str:
        """
        Convert the dataclass to a JSON string.
        """
        import json

        return json.dumps(self.to_dict(), indent=4)

    def to_yaml(self) -> str:
        """
        Convert the dataclass to a YAML string.
        """
        import yaml

        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

@unique
class ImageFilename(Enum):
    """
    Enum for image filename components.
    """

    SPECIES = "species"
    SEX = "sex"
    SITE = "collection_site"
    YEAR = "year"
    ID_IN_SET = "id_in_set"
    WING_SIDE = "wing_side"
    WING_TYPE = "wing_type"


class FilenameParser:
    """
    Class to parse filenames of the format defined by the ImageFilename enum.
    """
    def __init__(self, filename: str):
        """
        Initialize with a filename and parse it into its components.
        Add the 'extension' key to the parsed data.
        """
        self.filename = filename
        self.parsed_data = self._parse()

    def _parse(self) -> Dict:
        stem, ext = self.filename.rsplit(".", 1)
        parts = stem.split("_")

        if len(parts) != len(ImageFilename):
            raise ValueError(
                f"Filename '{self.filename}' does not match expected format."
            )

        parsed = {field.name.lower(): value for field, value in zip(ImageFilename, parts)}
        parsed["extension"] = ext.lower()
        return parsed

    def _generate_null(self) -> Dict:
        """
        Generate a dictionary with all fields set to None.
        """
        _null_record = {
            field.name.lower(): None for field in ImageFilename
        }
        _null_record["extension"] = None
        return _null_record
