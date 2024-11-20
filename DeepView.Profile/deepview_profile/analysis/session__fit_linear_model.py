def _fit_linear_model(x, y):
    y_np = np.array(y)
    x_np = np.array(x)
    stacked = np.vstack([x_np, np.ones(len(x_np))]).T
    slope, bias = np.linalg.lstsq(stacked, y_np, rcond=None)[0]
    return slope, bias
