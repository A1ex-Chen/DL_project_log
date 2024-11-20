def _check_img_dtype(img):
    assert isinstance(img, np.ndarray
        ), '[Augmentation] Needs an numpy array, but got a {}!'.format(type
        (img))
    assert not isinstance(img.dtype, np.integer
        ) or img.dtype == np.uint8, '[Augmentation] Got image of type {}, use uint8 or floating points instead!'.format(
        img.dtype)
    assert img.ndim in [2, 3], img.ndim
