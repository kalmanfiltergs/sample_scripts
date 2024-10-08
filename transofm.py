import numpy as np
import pandas as pd

def rolling_metric(
    df, 
    y_col,       # type: str
    x_col,       # type: str
    window,      # type: int
    metric_func, # type: callable
    min_window,  # type: int
    **kwargs
):
    """
    Compute a rolling metric over the DataFrame using a specified metric function.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - y_col (str): Name of the dependent variable column.
    - x_col (str): Name of the independent variable column.
    - window (int): Size of the rolling window.
    - metric_func (callable): Function to compute the metric over each window.
    - min_window (int): Minimum number of observations required to compute the metric.
    - **kwargs: Additional keyword arguments to pass to the metric function.

    Returns:
    - pd.Series: Rolling metric values aligned with the original DataFrame index.
    """
    y = df[y_col].values
    x = df[x_col].values
    n = len(y)

    if window < min_window:
        raise ValueError("Window size must be at least as large as min_window.")

    # Create rolling windows using stride tricks
    shape = (n - window + 1, window)
    strides = (y.strides[0], y.strides[0])

    y_roll = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    x_roll = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # Compute the metric over each window
    metric_values = metric_func(y_roll, x_roll, **kwargs)

    # Initialize the result with NaNs
    metric_series = pd.Series(data=np.nan, index=df.index)

    # Assign computed metric values where the window condition is met
    metric_series.iloc[window - 1:] = metric_values

    return metric_series

def wls_slope_no_intercept(
    y_roll,   # type: np.ndarray
    x_roll,   # type: np.ndarray
    weights    # type: np.ndarray
):
    """
    Compute the WLS slope without intercept over rolling windows.

    Parameters:
    - y_roll (np.ndarray): 2D array of rolling windows for y.
    - x_roll (np.ndarray): 2D array of rolling windows for x.
    - weights (np.ndarray): 1D array of weights.

    Returns:
    - np.ndarray: Array of WLS slope values.
    """
    # Normalize weights to sum to 1
    weights = weights / weights.sum()

    # Reshape weights for broadcasting
    w = weights.reshape(1, -1)

    # Calculate weighted sums
    weighted_xy = w * x_roll * y_roll
    weighted_xx = w * x_roll ** 2

    # Sum over the window
    numerator = weighted_xy.sum(axis=1)
    denominator = weighted_xx.sum(axis=1)

    # Calculate slope; handle division by zero
    slope = np.divide(numerator, denominator, 
                      out=np.full_like(numerator, np.nan), 
                      where=denominator != 0)

    return slope

def rolling_corr(
    y_roll, # type: np.ndarray
    x_roll  # type: np.ndarray
):
    """
    Compute the rolling correlation between y and x.

    Parameters:
    - y_roll (np.ndarray): 2D array of rolling windows for y.
    - x_roll (np.ndarray): 2D array of rolling windows for x.

    Returns:
    - np.ndarray: Array of correlation coefficients.
    """
    # Compute means
    y_mean = y_roll.mean(axis=1, keepdims=True)
    x_mean = x_roll.mean(axis=1, keepdims=True)

    # Demean the data
    y_demean = y_roll - y_mean
    x_demean = x_roll - x_mean

    # Compute numerator and denominator for correlation
    numerator = (y_demean * x_demean).sum(axis=1)
    denominator = np.sqrt((y_demean ** 2).sum(axis=1) * (x_demean ** 2).sum(axis=1))

    # Calculate correlation; handle division by zero
    corr = np.divide(numerator, denominator, 
                     out=np.full_like(numerator, np.nan), 
                     where=denominator != 0)

    return corr


