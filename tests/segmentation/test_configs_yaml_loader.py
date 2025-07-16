import pytest
from pathlib import Path
from hopperscapes.configs import SegmentationModelConfigs

YAML_FILEPATH = Path(__file__).parent.parent.parent / "hopperscapes" / "configs.yaml"


@pytest.mark.unit
def test_config_loader(debug=False):
    """
    Test configuration loading from YAML file.
    """
    yaml_path = YAML_FILEPATH

    configs = SegmentationModelConfigs.from_yaml(yaml_path)

    if debug:
        print("Model Name:", configs.model_name)
        print("Save Directory:", configs.savedir)
        print("Device:", configs.device)
        print("Square Image Size:", configs.square_image_size)
        print("Convert to HSV:", configs.convert_to_hsv)
        print("Out Channels:", configs.out_channels)
        print("Batch Size:", configs.batch_size)
        print("Learning Rate:", configs.learning_rate)
        print("Loss Function Configurations:", configs.loss_function_configs)
        print("Dice Scores to Track:", configs.dice_scores_to_track)
