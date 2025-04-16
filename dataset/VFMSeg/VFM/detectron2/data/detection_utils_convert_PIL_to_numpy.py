def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        conversion_format = format
        if format in ['BGR', 'YUV-BT.601']:
            conversion_format = 'RGB'
        image = image.convert(conversion_format)
    image = np.asarray(image)
    if format == 'L':
        image = np.expand_dims(image, -1)
    elif format == 'BGR':
        image = image[:, :, ::-1]
    elif format == 'YUV-BT.601':
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)
    return image
