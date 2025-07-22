import pytest
from pathlib import Path
from hopperscapes.configs import SegmentationModelConfigs

TRAIN_YAML_FILEPATH = (
    Path(__file__).parent.parent.parent / "configs" / "training" / "train.yaml"
)
DATASET_YAML_FILEPATH = (
    Path(__file__).parent.parent.parent / "configs" / "data" / "dataset.yaml"
)
MODEL_YAML_FILEPATH = (
    Path(__file__).parent.parent.parent / "configs" / "models" / "unet.yaml"
)

UNIFIED_YAML_FILEPATH = Path(__file__).parent.parent.parent / "configs" / "unified.yaml"


@pytest.mark.unit
def test_piecewise_configs_yaml_loader(debug=False):
    """
    Test configuration loading from YAML files.
    """
    configs = SegmentationModelConfigs.from_yaml_files(
        model_configs_path=MODEL_YAML_FILEPATH,
        dataset_configs_path=DATASET_YAML_FILEPATH,
        training_configs_path=TRAIN_YAML_FILEPATH,
    )


@pytest.mark.unit
def test_unifiedconfigs_yaml_loader(debug=False):
    """
    Test configuration loading from YAML file.
    """
    yaml_path = UNIFIED_YAML_FILEPATH

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
