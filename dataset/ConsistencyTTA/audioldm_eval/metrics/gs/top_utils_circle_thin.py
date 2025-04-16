def circle_thin(N=5000):
    phi = np.random.randn(N)
    x = [[np.sin(phi0), np.cos(phi0)] for phi0 in phi]
    x = np.array(x)
    x = x + 0.05 * np.random.randn(N, 2)
    return x
