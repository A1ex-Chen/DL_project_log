def preprocess_yolov6(image, channels=3, height=224, width=224):
    """Pre-processing for YOLOv6-based Object Detection Models

    Parameters
    ----------
    image: PIL.Image
        The image resulting from PIL.Image.open(filename) to preprocess
    channels: int
        The number of channels the image has (Usually 1 or 3)
    height: int
        The desired height of the image (usually 640)
    width: int
        The desired width of the image  (usually 640)

    Returns
    -------
    img_data: numpy array
        The preprocessed image data in the form of a numpy array

    """
    resized_image = image.resize((width, height), Image.BILINEAR)
    img_data = np.asarray(resized_image).astype(np.float32)
    if len(img_data.shape) == 2:
        img_data = np.stack([img_data] * 3)
        logger.debug('Received grayscale image. Reshaped to {:}'.format(
            img_data.shape))
    else:
        img_data = img_data.transpose([2, 0, 1])
    mean_vec = np.array([0.0, 0.0, 0.0])
    stddev_vec = np.array([1.0, 1.0, 1.0])
    assert img_data.shape[0] == channels
    for i in range(img_data.shape[0]):
        img_data[i, :, :] = (img_data[i, :, :] / 255.0 - mean_vec[i]
            ) / stddev_vec[i]
    return img_data
