def pil2cv(image: Image.Image) ->NDArray[np.uint8]:
    """Convert PIL image to OpenCV image.

    Args:
        image (Image.Image): PIL image

    Returns:
        NDArray[np.uint8]: Numpy Array (OpenCV image)
    """
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv.cvtColor(new_image, cv.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv.cvtColor(new_image, cv.COLOR_RGBA2BGRA)
    return new_image
