import numpy as np

def simulate_trades_numpy(time_series, bid, ask, req_edge, NAV1, NAV2):
    """
    Simulate trades based on the provided time series data and conditions, using numpy arrays.
    
    Parameters:
    - time_series (np.ndarray): Timestamps for each data point in the day.
    - bid (np.ndarray): Bid prices for the symbol.
    - ask (np.ndarray): Ask prices for the symbol.
    - req_edge (float): Required edge to consider a trade.
    - NAV1 (np.ndarray): NAV1 values over time.
    - NAV2 (np.ndarray): NAV2 values over time.
    
    Returns:
    - trades (list of dicts): List of trading opportunities with relevant information.
    """
    trades = []  # List to store trade information

    # Iterate over the time series to check for trading opportunities
    for i in range(len(time_series)):
        p_sell = NAV1[i] + req_edge  # Sell price threshold
        p_buy = NAV2[i] - req_edge   # Buy price threshold

        # Check if there's a selling opportunity (p_sell <= ask - 0.01)
        if p_sell <= ask[i] - 0.01:
            trades.append({
                'time': time_series[i],
                'direction': -1,  # Sell
                'bid': bid[i],
                'ask': ask[i],
                'p_sell': p_sell,
                'p_buy': p_buy,
                'NAV1': NAV1[i],
                'NAV2': NAV2[i]
            })

        # Check if there's a buying opportunity (p_buy >= bid + 0.01)
        if p_buy >= bid[i] + 0.01:
            trades.append({
                'time': time_series[i],
                'direction': 1,  # Buy
                'bid': bid[i],
                'ask': ask[i],
                'p_sell': p_sell,
                'p_buy': p_buy,
                'NAV1': NAV1[i],
                'NAV2': NAV2[i]
            })

    return trades

# Example data (numpy arrays)
time_series = np.array(['09:30', '09:31', '09:32'])
bid = np.array([100.5, 100.6, 100.7])
ask = np.array([101.0, 101.1, 101.2])
NAV1 = np.array([100.8, 100.9, 101.0])
NAV2 = np.array([100.2, 100.3, 100.4])
req_edge = 0.02

# Call the function with numpy arrays
trades = simulate_trades_numpy(time_series, bid, ask, req_edge, NAV1, NAV2)

# Example output
for trade in trades:
    print(trade)
