import numpy as np
import pandas as pd

def rolling_metric(df, y_col, x_col, window, metric_func, **kwargs):
    y = df[y_col].values
    x = df[x_col].values
    n = len(y)

    # Create rolling windows
    shape = (n - window + 1, window)
    strides = y.strides * 2
    y_roll = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    x_roll = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # Compute metric
    metric_values = metric_func(y_roll, x_roll, **kwargs)

    # Align results with original DataFrame
    metric_series = pd.Series(np.nan, index=df.index)
    metric_series.iloc[window - 1:] = metric_values

    return metric_series

def wls_slope_no_intercept(y_roll, x_roll, weights):
    weights = weights / weights.sum()
    w = weights.reshape(1, -1)
    numerator = (w * x_roll * y_roll).sum(axis=1)
    denominator = (w * x_roll**2).sum(axis=1)
    slope = np.divide(numerator, denominator, where=denominator != 0)
    return slope

def rolling_corr(y_roll, x_roll):
    y_mean = y_roll.mean(axis=1, keepdims=True)
    x_mean = x_roll.mean(axis=1, keepdims=True)
    y_demean = y_roll - y_mean
    x_demean = x_roll - x_mean
    numerator = (y_demean * x_demean).sum(axis=1)
    denominator = np.sqrt((y_demean**2).sum(axis=1) * (x_demean**2).sum(axis=1))
    corr = np.divide(numerator, denominator, where=denominator != 0)
    return corr

def linear_weights(window):
    return np.arange(1, window + 1)

def exponential_weights(window, alpha=0.1):
    return np.exp(alpha * np.arange(window))

# Example DataFrame
np.random.seed(0)
df = pd.DataFrame({
    'col1': np.random.randn(1000).cumsum(),  # Dependent variable
    'col2': np.random.randn(1000).cumsum()   # Independent variable
})

window_size = 100
weights = exponential_weights(window_size, alpha=0.1)

# Rolling WLS coefficient (no intercept)
df['wls_coef'] = rolling_metric(
    df,
    y_col='col1',
    x_col='col2',
    window=window_size,
    metric_func=wls_slope_no_intercept,
    weights=weights
)

# Rolling correlation
df['rolling_corr'] = rolling_metric(
    df,
    y_col='col1',
    x_col='col2',
    window=window_size,
    metric_func=rolling_corr
)



def rolling_wls_slope(df, y_col, x_col, window, weights=None):
    """
    Compute the rolling Weighted Least Squares (WLS) slope without an intercept.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - y_col (str): Name of the dependent variable column.
    - x_col (str): Name of the independent variable column.
    - window (int): Size of the rolling window.
    - weights (array-like, optional): Weights for the WLS. If None, equal weights are used.

    Returns:
    - pd.Series: Rolling WLS slope values aligned with the original DataFrame index.
    """
    y = df[y_col].values
    x = df[x_col].values
    n = len(y)

    # Handle weights
    if weights is None:
        weights = np.ones(window)
    else:
        weights = np.array(weights)
        if len(weights) != window:
            raise ValueError("Length of weights must equal the window size.")
        weights = weights / weights.sum()  # Normalize weights

    # Use stride tricks to create rolling windows
    shape = (n - window + 1, window)
    strides = (y.strides[0], y.strides[0])
    
    y_roll = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    x_roll = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # Compute weighted sums
    weighted_xy = weights * x_roll * y_roll
    weighted_xx = weights * x_roll ** 2

    # Sum over the window
    numerator = weighted_xy.sum(axis=1)
    denominator = weighted_xx.sum(axis=1)

    # Calculate slope; handle division by zero
    slope = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator!=0)

    # Create a Series with NaNs for the initial periods
    slope_series = pd.Series(data=np.nan, index=df.index)
    slope_series.iloc[window-1:] = slope

    return slope_series

def linear_weights(window):
    return np.arange(1, window + 1)

def exponential_weights(window, alpha=0.1):
    return np.exp(alpha * np.arange(window))


# Example DataFrame
np.random.seed(0)
window_size = 100
data_size = 1000
df = pd.DataFrame({
    'col1': np.random.randn(data_size).cumsum(),  # Simulated price changes
    'col2': np.random.randn(data_size).cumsum()   # Simulated volume changes
})

# Define weights
weights_linear = linear_weights(window_size)
weights_exponential = exponential_weights(window_size, alpha=0.1)

# Compute rolling WLS slope with linear weights
df['wls_slope_linear'] = rolling_wls_slope(df, 'col1', 'col2', window=window_size, weights=weights_linear)

# Compute rolling WLS slope with exponential weights
df['wls_slope_exponential'] = rolling_wls_slope(df, 'col1', 'col2', window=window_size, weights=weights_exponential)

# Optional: Compare with rolling correlation
df['rolling_corr'] = df['col1'].rolling(window=window_size).corr(df['col2'])
