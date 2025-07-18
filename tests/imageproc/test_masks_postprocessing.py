import pytest


@pytest.mark.unit
def test_pick_largest_region():
    from hopperscapes.imageproc.masks import pick_largest_region
    import numpy as np

    mask = np.zeros((256, 256), dtype=bool)
    mask[50:150, 50:150] = True  # large region
    expected_area = np.sum(mask)

    mask[32:48, 32:48] = True  # small region

    result = pick_largest_region(mask)
    assert result.shape == mask.shape
    assert np.sum(result) == expected_area


@pytest.mark.unit
def test_seeded_watershed():
    from hopperscapes.imageproc.masks import seeded_watershed
    import numpy as np

    image = np.zeros((256, 256), dtype=np.uint8)
    image[50:100, 50:100] = 255
    image[150:200, 150:200] = 255

    seed_mask = np.zeros((256, 256), dtype=bool)
    seed_mask[80, 80] = True
    seed_mask[180, 180] = True

    # Test the function
    labels = seeded_watershed(image, seed_mask)

    assert len(np.unique(labels)) - 1 == 2  # two regions + background


@pytest.mark.unit
def test_seeded_watershed_with_local_maxima():
    from hopperscapes.imageproc.masks import seeded_watershed_local_maxima
    import numpy as np

    # overlapping masks
    seeds_mask = np.zeros((256, 256), dtype=bool)
    seeds_mask[50:100, 50:100] = True
    seeds_mask[80:180, 80:180] = True

    labels = seeded_watershed_local_maxima(
        seeds_mask, min_distance=10, exclude_border=False
    )

    assert len(np.unique(labels)) - 1 == 2  # two regions + background
