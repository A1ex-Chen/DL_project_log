def assert_mean_pixel_difference(image, expected_image, expected_max_diff=10):
    image = np.asarray(DiffusionPipeline.numpy_to_pil(image)[0], dtype=np.
        float32)
    expected_image = np.asarray(DiffusionPipeline.numpy_to_pil(
        expected_image)[0], dtype=np.float32)
    avg_diff = np.abs(image - expected_image).mean()
    assert avg_diff < expected_max_diff, f'Error image deviates {avg_diff} pixels on average'
