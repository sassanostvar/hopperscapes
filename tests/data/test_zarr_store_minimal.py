import pytest


@pytest.mark.unit
def test_zarr_store_class_minimal():
    """
    Test the zarr store generator class constructer
    """
    from hopperscapes.data.zarr_store import (
        HopperWingsZarrStore,
        HopperWingsZarrStoreConfig,
    )

    # test init class without config (uses default configs)
    _ = HopperWingsZarrStore()

    # test init class with config
    configs = HopperWingsZarrStoreConfig()
    _ = HopperWingsZarrStore(config=configs)
