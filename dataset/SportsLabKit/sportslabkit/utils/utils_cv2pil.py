def cv2pil(image: NDArray[np.uint8], convert_bgr2rgb=True) ->Image.Image:
    """Convert OpenCV image to PIL image.

    Args:
        image (NDArray[np.uint8]): Numpy Array (OpenCV image)

    Returns:
        Image.Image: PIL image
    """
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        if convert_bgr2rgb:
            new_image = cv.cvtColor(new_image, cv.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:
        if convert_bgr2rgb:
            new_image = cv.cvtColor(new_image, cv.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image
