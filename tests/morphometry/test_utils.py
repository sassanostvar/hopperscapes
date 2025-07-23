import pytest 

@pytest.mark.unit
def test_filter_outliers():
    import numpy as np
    from hopperscapes.morphometry.utils import remove_outliers_iqr

    data = np.array([1, 2, 3., np.pi, 4, 5, 100])  # 100 is an outlier
    filtered_data = remove_outliers_iqr(data)
    assert len(filtered_data) == 6  # 100 should be removed