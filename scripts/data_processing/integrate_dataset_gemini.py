"""
Inventory-only pass: build a per-image CSV that merges a specimen spreadsheet
with on-disk wing images.  No files are written into the image tree; images are
left untouched.

Expected directory layout
-------------------------
<image_root>/<year>/<mm-dd-yy>/<street_address>/<specimen_folder>/*.jpg

Filename pattern (case-insensitive)
```
<genotype>_<sex>[MF]_<city>_<five-digit-year>_<id4>_<left|right>_forewing[rep].jpg
```
If multiple replicates exist, the first left and first right image per specimen
are kept; extras are logged and skipped.

Spreadsheet requirements
-----------------------
CSV with at least these columns (exact spellings):
```
SPECIMEN_ID, CONTAINER_TYPE, DATE_COLLECTED, LOC, SEX, PRESERV_MEDIUM,
DRIED, IMAGING_INSTRUMENT, DATE_IMAGED, MAG, OBJ, DAMAGED, SUCCESS, NOTES
```
`SPECIMEN_ID` must equal  ``<genotype>_<city>_<year>_<id>``  (sex *omitted*).

Outputs
-------
* *integrated CSV* (per-image rows) - path given by `--out-csv`
* *log file* - path given by `--log`

No side-car metadata files are produced in this phase.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from dataclasses import dataclass

from PIL import Image, ExifTags

# -----------------------------------------------------------------------------
# project imports - assumes src.record.Metadata is importable via PYTHONPATH
# -----------------------------------------------------------------------------
try:
    from hopperscapes.data.record import (
        ContainerType,
        ImagingInstrument,
        Metadata,
        PreserveMedium,
        WingSide,
        WingType,
    )
except ImportError as e:  # pragma: no cover
    logging.critical("Cannot import src.record.* - adjust PYTHONPATH")
    raise e

# -----------------------------------------------------------------------------
logger = logging.getLogger("integrate_dataset")

FILENAME_RE = re.compile(
    r"^(?P<genotype>[^_]+)_(?P<sex>[MF])_"
    r"(?P<city>[^_]+)_(?P<year>\d{5})_"
    r"(?P<id>\d{4})_"
    r"(?P<side>left|right)_(?P<wing>forewing)(?P<rep>\d*)$",
    re.IGNORECASE,
)

# DATE_FMT_IN = "%m-%d-%y"  # e.g. 8-30-24
DATE_FMT_IN = "%Y-%m-%d"  # e.g. 2024-8-30

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def sha256_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_filename(stem: str) -> Dict[str, str] | None:
    m = FILENAME_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()
    d["rep"] = int(d["rep"]) if d["rep"] else 1
    return d


def standardize_date(s: str) -> str | None:
    """Standardize date string to YYYY-MM-DD format."""
    if not isinstance(s, str) or not s.strip():
        return None
    # collapse one or more '/' or '-' into a single '-'
    s = re.sub(r"[/-]+", "-", s.strip())
    return s


def parse_date(s: str | float | None) -> date | None:
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        s_std = standardize_date(s)
        return datetime.strptime(s_std.strip(), DATE_FMT_IN).date()
    except ValueError:
        logger.warning(f"Date '{s}' could not be parsed into MM-DD-YY; storing None.")
        return None


def date_to_str(d: date | None) -> str | None:
    """Convert date to string in YYYY-MM-DD format."""
    if d is None:
        return None
    return d.strftime("%Y-%m-%d")

def str_bool(x: str | float | None) -> bool | None:
    if isinstance(x, str):
        if x.strip().upper() in {"Y", "YES", "TRUE"}:
            return True
        if x.strip().upper() in {"N", "NO", "FALSE"}:
            return False
    return None


def map_container(s: str | None) -> ContainerType:
    if not isinstance(s, str):
        return ContainerType.UNKNOWN
    s_lower = s.lower()
    if "50" in s_lower:
        return ContainerType.GREINER_CONICAL_50ML
    if "15" in s_lower:
        return ContainerType.UNKNOWN_CONICAL_15ML
    return ContainerType.UNKNOWN


def map_preserve(s: str | None) -> PreserveMedium:
    if not isinstance(s, str):
        return PreserveMedium.UNKNOWN
    s_up = s.strip().upper()
    if "95" in s_up:
        return PreserveMedium.ETHANOL_95
    if "70" in s_up:
        return PreserveMedium.ETHANOL_70
    return PreserveMedium.UNKNOWN

def map_imaging_instrument(s: str | None) -> ImagingInstrument:
    if not isinstance(s, str):
        return ImagingInstrument.UNKNOWN
    s_up = s.strip().upper()
    if "AMSCOPE" in s_up:
        return ImagingInstrument.AMSCOPE
    if "LEICA" in s_up:
        return ImagingInstrument.LEICA
    return ImagingInstrument.UNKNOWN


# -----------------------------------------------------------------------------
# Helpers to read image metadata
# -----------------------------------------------------------------------------

@dataclass
class ImageHeader:
    shape: Tuple[int, int, int]  # (height, width, channels)
    dpi: Optional[Tuple[int, int]] = None
    pixel_size_um: Optional[Tuple[float, float]] = None
    color_space: Optional[str] = None
    color_profile: Optional[str] = None


def convert_pixel_size_um(
        res: float, 
        unit: int):
    if unit == 2: # inches
        return 25_400 / res  # convert DPI to microns
    elif unit == 3: # centimeters
        return 1_000_000 / res  # convert DPCM to microns
    else:
        raise ValueError(f"Unsupported unit: {unit}. Expected 2 (inches) or 3 (centimeters).")


def read_image_header(img_path: Path) -> ImageHeader:
    """Read image header only without loading the full image."""
    with Image.open(img_path) as img:
        y,x = img.size[::-1]  # PIL uses (width, height) order
        dpi=img.info.get("dpi") or img.info.get("resolution")
        color_space = img.mode # e.g. "RGB", "HSV"
        color_profile = img.info.get("icc_profile", None)
        channels = 3 if color_space in {"RGB", "HSV"} else 1
        if dpi:
            pixel_size_um = (
                convert_pixel_size_um(dpi[0], unit=2), # inches
                convert_pixel_size_um(dpi[1], unit=2)  # inches
            )
        else:
            # try exif 
            exif = img.getexif()
            if exif:
                x_res = exif.get(ExifTags.TAGS.get('XResolution', None))
                y_res = exif.get(ExifTags.TAGS.get('YResolution', None))
                unit = exif.get(ExifTags.TAGS.get('ResolutionUnit', None))
                if x_res and y_res and unit:
                    pixel_size_um = (
                        convert_pixel_size_um(x_res, unit),
                        convert_pixel_size_um(y_res, unit)
                    )
                else:
                    pixel_size_um = None
            else:
                pixel_size_um = None

        return ImageHeader(
            shape=(y, x, channels),
            dpi=dpi,
            pixel_size_um=pixel_size_um,
            color_space=color_space,
            color_profile=color_profile,
        )

# -----------------------------------------------------------------------------
class Integrator:
    def __init__(
        self,
        image_root: Path,
        spreadsheet_csv: Path,
        out_csv: Path,
        merge_with_input: bool = False,
    ) -> None:
        self.image_root = image_root
        self.out_csv = out_csv
        self.merge_with_input = merge_with_input

        # Spreadsheet load & index by SPECIMEN_ID
        df = pd.read_csv(spreadsheet_csv, dtype=str).fillna("")
        if "SPECIMEN_ID" not in df.columns:
            raise KeyError("Spreadsheet missing SPECIMEN_ID column")
        self.df_meta = df.set_index("SPECIMEN_ID")

        # specimen_id -> {"left": Path, "right": Path}
        self.chosen: Dict[str, Dict[str, Path]] = {}
        self.rows: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    def walk_images(self) -> None:
        for img_path in self.image_root.rglob("*.jpg"):
            tokens = parse_filename(img_path.stem)
            if tokens is None:
                logger.warning(f"Bad filename: {img_path.relative_to(self.image_root)}")
                continue

            specimen_id = (
                f"{tokens['genotype']}_{tokens['city']}_{tokens['year']}_{tokens['id']}"
            )
            side = tokens["side"].lower()
            slot = self.chosen.setdefault(specimen_id, {})
            if side in slot:
                logger.info(f"Skip replicate: {img_path.relative_to(self.image_root)}")
                continue
            slot[side] = img_path

    # ------------------------------------------------------------------
    def build_rows(self) -> None:
        for specimen_id, sides in self.chosen.items():
            try:
                row = self.df_meta.loc[specimen_id]
            except KeyError:
                logger.warning(f"Specimen {specimen_id} not in spreadsheet; skipped")
                continue

            for side, img_path in sides.items():
                tokens = parse_filename(img_path.stem)
                if not tokens:
                    continue  # shouldn't happen

                md = self.compose_metadata(tokens, side, img_path, row)

                if self.merge_with_input:
                    merged = row.to_dict()
                    merged.update(md.to_dict())
                    merged["filepath"] = str(img_path)
                    self.rows.append(merged)
                else:
                    # Only keep the metadata
                    md_dict = md.to_dict()
                    md_dict["filepath"] = str(img_path)
                    self.rows.append(md_dict)

    # ------------------------------------------------------------------
    def compose_metadata(
        self, tokens: Dict[str, str], side: str, img_path: Path, row: pd.Series
    ) -> Metadata:
        """Compose Metadata object from filename tokens and spreadsheet row."""
        img_header = read_image_header(img_path)
        return Metadata(
            record_id=img_path.stem,
            specimen_id=row.name,
            species=tokens["genotype"],
            year=int(tokens["year"].lstrip("0")),
            date_collected=date_to_str(parse_date(row.get("DATE_COLLECTED"))),
            site_code=row.get("LOC") or None,
            coordinates=None,
            collected_by=None,
            live_sample=None,
            imaged_by=None,
            sex=row.get("SEX") or tokens["sex"],
            preserve_medium=map_preserve(row.get("PRESERV_MEDIUM")),
            dried=row.get("DRIED") or None,
            container_type={map_container(row.get("CONTAINER_TYPE"))},
            imaging_instrument=map_imaging_instrument(row.get("IMAGING_INSTRUMENT")),
            date_imaged=date_to_str(parse_date(row.get("DATE_IMAGED"))),
            mag=row.get("MAG") or None,
            obj=row.get("OBJ") or None,
            damaged=str_bool(row.get("DAMAGED")),
            image_success=str_bool(row.get("SUCCESS")),
            discarded=None,
            notes=row.get("NOTES") or None,
            wing_type=WingType.FOREWING,
            wing_side=WingSide(side),
            replicate_number=tokens["rep"],
            best_in_set=(tokens["rep"] == 1),
            image_size=img_header.shape,
            image_channels=None,
            image_bit_depth=None,
            image_file_ext=img_path.suffix.lstrip("."),
            color_space=img_header.color_space,
            color_profile=img_header.color_profile,
            image_resolution=img_header.dpi,
            pixel_size_um=img_header.pixel_size_um,
            file_hash=sha256_hash(img_path),
        )

    # ------------------------------------------------------------------
    def write_csv(self) -> None:
        if not self.rows:
            logger.error("No valid image rows collected - CSV not written")
            return
        pd.DataFrame(self.rows).to_csv(self.out_csv, index=False)
        logger.info(f"Wrote {len(self.rows)} rows to {self.out_csv}")

    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info("Scanning image tree...")
        self.walk_images()
        logger.info(
            f"Selected {sum(len(v) for v in self.chosen.values())} images from {len(self.chosen)} specimens"
        )
        self.build_rows()
        self.write_csv()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate inventory CSV for SLF wing images"
    )
    parser.add_argument("--image-root-path", required=True, type=Path)
    parser.add_argument("--table-path", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--log", type=Path, default=Path("integrate_dataset.log"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.log, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    Integrator(
        image_root=args.image_root_path,
        spreadsheet_csv=args.table_path,
        out_csv=args.out_csv,
    ).run()


if __name__ == "__main__":
    main()
