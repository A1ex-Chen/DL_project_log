def imread(filename: str, flags: int=cv2.IMREAD_COLOR):
    """
    Read an image from a file.

    Args:
        filename (str): Path to the file to read.
        flags (int, optional): Flag that can take values of cv2.IMREAD_*. Defaults to cv2.IMREAD_COLOR.

    Returns:
        (np.ndarray): The read image.
    """
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)
