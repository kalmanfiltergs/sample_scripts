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
