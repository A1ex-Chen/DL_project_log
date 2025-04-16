@staticmethod
def iterative_sigma_clipping(data, sigma=2, max_iters=3):
    """Applies an iterative sigma clipping algorithm to the given data times number of iterations."""
    data = np.array(data)
    for _ in range(max_iters):
        mean, std = np.mean(data), np.std(data)
        clipped_data = data[(data > mean - sigma * std) & (data < mean + 
            sigma * std)]
        if len(clipped_data) == len(data):
            break
        data = clipped_data
    return data
