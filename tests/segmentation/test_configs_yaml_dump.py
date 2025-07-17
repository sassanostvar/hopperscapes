import pytest
from pathlib import Path

YAML_FILEPATH = (
    Path(__file__).parent.parent.parent / "outputs" / "test_outputs" / "configs.yaml"
)


@pytest.mark.unit
def test_configs_yaml_writer():
    """
    Test configuration loading from YAML file.
    """
    from hopperscapes.configs import SegmentationModelConfigs

    parent_dir = YAML_FILEPATH.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    configs = SegmentationModelConfigs()
    configs.to_yaml(YAML_FILEPATH)
