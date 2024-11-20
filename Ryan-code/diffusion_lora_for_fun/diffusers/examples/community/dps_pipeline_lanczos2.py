def lanczos2(x):
    return (np.sin(pi * x) * np.sin(pi * x / 2) + np.finfo(np.float32).eps) / (
        pi ** 2 * x ** 2 / 2 + np.finfo(np.float32).eps) * (abs(x) < 2)
