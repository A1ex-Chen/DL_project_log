def generate_distortion_fn(distortion_name, severity):
    aug_fn = DISTORTION_REGISTRY.get(distortion_name)

    def distort_batch(x):
        x = np.asarray(x, dtype=np.uint8)
        x = aug_fn(x, severity=severity)
        return PILImage.fromarray(np.uint8(x))
    return distort_batch
