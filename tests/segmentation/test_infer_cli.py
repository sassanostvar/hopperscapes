import pytest 
from pathlib import Path
import subprocess

CHECKPOINT_PATH = (
    Path(__file__).parent.parent.parent / "checkpoints" / "HopperNetLite_demo.pth"
)

CONFIGS_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "unified_lite.yaml"
)

SAMPLE_IMAGE_PATH = (
    Path(__file__).parent.parent / "test_data" / "LD_F_TC_02024_0024_left_forewing.jpg"
)

@pytest.mark.unit
def test_infer_cli():
    result = subprocess.run([
        "python", "-m", "hopperscapes.segmentation.infer",
        "--image_path", str(SAMPLE_IMAGE_PATH),
        "--checkpoint_path", str(CHECKPOINT_PATH),
        "--configs_path", str(CONFIGS_PATH),
        "--output_dir", "./outputs/test_infer_cli",
        "--overwrite",
        "--device", "cpu",
    ])

    assert result.returncode == 0, f"Inference CLI failed: {result.stderr}"