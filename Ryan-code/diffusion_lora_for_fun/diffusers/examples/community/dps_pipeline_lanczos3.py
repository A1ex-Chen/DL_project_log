def lanczos3(x):
    return (np.sin(pi * x) * np.sin(pi * x / 3) + np.finfo(np.float32).eps) / (
        pi ** 2 * x ** 2 / 3 + np.finfo(np.float32).eps) * (abs(x) < 3)
