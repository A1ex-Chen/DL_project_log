def filled_circle(N=5000):
    ans = []
    while len(ans) < N:
        x = np.random.rand(2) * 2.0 - 1.0
        if np.linalg.norm(x) < 1:
            ans.append(x)
    return np.array(ans) + 0.05 * np.random.randn(N, 2)
