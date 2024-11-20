def distort_batch(x):
    x = np.asarray(x, dtype=np.uint8)
    x = aug_fn(x, severity=severity)
    return PILImage.fromarray(np.uint8(x))
