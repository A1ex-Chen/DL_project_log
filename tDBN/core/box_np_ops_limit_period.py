def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period
