"""
Utility functions for morphometric analysis.
"""

from numpy.typing import NDArray


def remove_outliers_iqr(data: NDArray, threshold: float = 1.5) -> NDArray:
    """
    Remove outliers from the data using the Interquartile Range (IQR) method.
    Args:
        data (NDArray): Input data array.
        threshold (float): IQR multiplier to define outliers (default is 1.5).
    Returns:
        NDArray: Data array with outliers removed.
    """

    import numpy as np

    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]
