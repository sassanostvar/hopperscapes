import pandas as pd
import pytest


@pytest.mark.unit
def test_record_dataclass_minimal():
    """
    Instantiate the metadata dataclass and test the conversion methods.
    """
    from hopperscapes.data.record import Metadata

    metadata_obj = Metadata()

    # cast to dict
    _dict = metadata_obj.to_dict()
    assert isinstance(_dict, dict)

    # cast to pandas df
    _df = metadata_obj.to_pandas()
    assert isinstance(_df, pd.DataFrame)

    # cast to yaml
    _yaml = metadata_obj.to_yaml()
    assert isinstance(_yaml, str)
