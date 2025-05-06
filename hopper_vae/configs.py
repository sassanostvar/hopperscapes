from dataclasses import dataclass


@dataclass
class SegmentationConfig:
    """
    Configuration for the multi-head segmentation model.
    """
    heads = {
        "wing",
        "veins",
        "spots",
        "domains",
        # "bricks",
    }
    square_image_size = 512
