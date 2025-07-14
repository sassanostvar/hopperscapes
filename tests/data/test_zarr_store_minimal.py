import pytest


@pytest.mark.unit
def test_zarr_store_class_minimal():
    """
    Test the zarr store generator class constructer
    """
    from hopper_vae.data.zarr_store import (
        HopperWingsZarrStore,
        HopperWingsZarrStoreConfig,
    )

    # test init class without config (uses default configs)
    zstore = HopperWingsZarrStore()

    # test init class with config
    configs = HopperWingsZarrStoreConfig()
    zstore = HopperWingsZarrStore(config=configs)
