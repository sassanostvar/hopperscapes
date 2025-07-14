from pathlib import Path

import pytest

IMAGE_PATH = (
    Path(__file__).parent.parent / "test_data" / "LD_F_TC_02024_0024_left_forewing.jpg"
)


@pytest.mark.unit
def test_read_image_header():
    from PIL import ExifTags, Image

    with Image.open(IMAGE_PATH) as img:
        info_keys = img.info.keys()
        for key in info_keys:
            print(f"{key}: {img.info[key]}")

        print(f"image mode: {img.mode}")

        exif = img.getexif()
        if exif:
            for tag, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                print(f"{tag_name}: {value}")
