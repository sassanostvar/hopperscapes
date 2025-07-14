"""
Build an OME-Zarr store from tabulated paths to (paired) forewing images 
and associated metadata.

Layout
------
<store>.zarr/
└─ <specimenID>/
   └─ forewing/
      ├─ left/
      │   ├─ rgb/  (3 x Y x X multiscale)
      │   └─ .attrs
      └─ right/
          ├─ rgb/
          └─ .attrs

Metadata are defined in hopper_vae/data/record.py and are stored in the
Zarr store as attributes of the group corresponding to the image.
"""


import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict

import dask.array as da
import numcodecs
import pandas as pd
import zarr
from dask import delayed
from dask.diagnostics import ProgressBar
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from skimage import io
from tqdm import tqdm

logger = logging.getLogger(__name__)

SYNC = zarr.ThreadSynchronizer()


@dataclass
class HopperWingsZarrStoreConfig:
    """
    Configuration for the Hopper Wings Zarr store.
    """

    zarr_name: str = "hopper_wings_raw"
    path: Path = Path(f"./data/{zarr_name}.zarr")
    rewrite: bool = False
    n_workers: int = 4
    size_xy: int = 1024
    size_z: int = 1
    size_c: int = 3
    levels: int = 2
    scale: list = field(default_factory=lambda: [[1, 2, 2]] * 2)
    compression: str = "zstd"
    compression_level: int = 4


class HopperWingsZarrStore:
    """
    Class to create Zarr store for the Hopper Wings dataset.
    """

    default_config = HopperWingsZarrStoreConfig()

    def __init__(
        self,
        config: HopperWingsZarrStoreConfig = None,
        records_table: pd.DataFrame = None,
    ):
        self.path = config.path
        self.root = None
        self.config = config if config else self.default_config

        self.records_table = records_table
        self.records = {}

        self.overwrite = self.config.rewrite

    def _make_compressor(self):
        """Return a ready-to-use numcodecs compressor."""
        return numcodecs.get_codec(
            {"id": self.config.compression, "level": self.config.compression_level}
        )

    def _create_base_zarr_store(self, path: Path) -> None:
        """
        Create a Zarr store for the Hopper Wings dataset.
        """
        # Create the Zarr store directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)

        # Create a Zarr group
        temp_root = zarr.group(
            store=parse_url(str(path), mode="w").store,
            overwrite=self.overwrite,
            synchronizer=SYNC,  # Use the global synchronizer
        )

        logging.info(f"Zarr store created at {path}")

        self.zarr_store_path = path

        # Create the main groups for specimens, wing types, and sides
        for _, augmented_row in tqdm(
            self.records_table.iterrows(), desc="Creating groups"
        ):
            # remove filepath from row
            row = augmented_row.copy()
            row.pop("filepath")
            _metadata = row.to_dict()

            # extract metadata fields used to create the group structure
            _specimen_id = _metadata["specimen_id"]
            _wing_type = _metadata["wing_type"]
            _side = _metadata["wing_side"]
            _record_id = _metadata["record_id"]

            #
            if _record_id not in self.records:
                self.records[_record_id] = augmented_row

            # create per-wing-side group structure
            record_grp = (
                (temp_root.require_group(_specimen_id))
                .require_group(_wing_type)
                .require_group(_side, overwrite=True)
            )

        logging.info(
            f"Zarr store structure created at {self.path} with {len(self.records)} records."
        )

    def _add_image_to_zarr(
        self, zarr_store_path: Path, record: Dict[str, Dict]
    ) -> None:
        """
        Add an image to the Zarr store.
        """

        if record is None or not isinstance(record, dict):
            logging.error("Record is None or not a dictionary.")
            return

        ((record_id, _metadata),) = record.items()

        specimen_id = _metadata.get("specimen_id", "unknown")
        wing_type = _metadata.get("wing_type", "unknown")
        wing_side = _metadata.get("wing_side", "unknown")
        filepath = _metadata["filepath"]

        # Read the image
        rgb_image = io.imread(filepath)

        if rgb_image.ndim == 2:  # Grayscale image
            raise ValueError("Image must be RGB or HSV, not grayscale.")

        # Get the group for this record
        current_root = zarr.group(
            store=parse_url(zarr_store_path, mode="a").store,
            overwrite=False,
            synchronizer=SYNC,  # Use the global synchronizer
        )

        record_grp = current_root[specimen_id][wing_type][wing_side]

        # add metadata
        _metadata.pop("filepath")
        serialisable = {
            k: (v.isoformat() if isinstance(v, (date, datetime)) else v)
            for k, v in _metadata.items()
        }
        record_grp.attrs.update(serialisable)

        for tag, arr, dtype in (("rgb", rgb_image.transpose(2, 0, 1), "uint8"),):
            comp = self._make_compressor()

            storage_opts = dict(
                compressor=comp,
                chunks=(1, self.config.size_xy, self.config.size_xy),  # same as before
            )

            write_image(
                arr,
                record_grp.require_group(tag),
                axes="cyx",
                channel_names=list(tag.upper()),
                dtype=dtype,
                scale_factors=self.config.scale,
                storage_options=storage_opts,
            )

            logging.info(
                f"Added {tag} image for {specimen_id}/{wing_type}/{wing_side} to Zarr store."
            )

    def run(self) -> None:
        """
        Run the Zarr store creation process.
        """
        if not self.path.exists() or self.overwrite:
            self._create_base_zarr_store(self.path)
        else:
            logging.info(
                f"Zarr store already exists at {self.path}. Use --rewrite to overwrite."
            )
            return

        if self.records_table is None or self.records_table.empty:
            logging.warning(
                "No records table provided or it is empty. Cannot add images to Zarr store."
            )
            return

        # Ensure the records table is a DataFrame
        if not isinstance(self.records_table, pd.DataFrame):
            raise TypeError("records_table must be a pandas DataFrame.")

        # Run image pyramide creation tasks
        tasks = [
            delayed(self._add_image_to_zarr)(str(self.path), {record_id: metadata})
            for record_id, metadata in self.records.items()  # Iterate through the correctly populated records
        ]

        with ProgressBar():
            da.compute(*tasks, scheduler="threads", num_workers=self.config.n_workers)

        logging.info(
            f"processed {len(self.records)} records and added images to Zarr store."
        )

        # consolidate metadata
        logging.info(f"consolidating metadata for {self.path}")
        store = parse_url(str(self.path), mode="a").store
        root_grp = zarr.open_group(store)
        zarr.consolidate_metadata(root_grp)
        logging.info("metadata consolidation complete.")


def main(args: list[str] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create Zarr stores for the Hopper Wings dataset."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./data/hopper_wings_raw.zarr",
        help="Path to the Zarr store directory.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size for the images.",
    )
    parser.add_argument(
        "--images-csv",
        type=str,
        default=None,
        help="Path to the CSV file containing image metadata and filepaths.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of worker processes to use.",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Rewrite the Zarr store if it already exists.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="./zarr_store_creation.log",
        help="Path to the log file for the Zarr store creation.",
    )

    args = parser.parse_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.log, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    config = HopperWingsZarrStoreConfig()

    config.path = Path(args.path)
    config.size_xy = args.chunk_size
    config.rewrite = args.rewrite
    config.n_workers = args.n_workers

    if args.images_csv:
        # Load the records table from the CSV file
        records_table = pd.read_csv(args.images_csv)
        if records_table.empty:
            logging.warning("No records found in the provided CSV file.")
    else:
        raise ValueError(
            "No images CSV file provided. Use --images-csv to specify one."
        )

    # Create the Zarr store
    zarr_processor = HopperWingsZarrStore(
        config=config,
        records_table=records_table,
    )
    zarr_processor.run()


if __name__ == "__main__":
    main()
