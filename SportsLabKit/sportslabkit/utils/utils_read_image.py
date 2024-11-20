def read_image(img):
    """Reads an image from a file, URL, a numpy array, or a torch tensor.
    Args:
        img (str, Path, Image.Image, np.ndarray, or torch.Tensor): The image to read.
    Returns:
        np.ndarray: The image as a numpy array.
    """
    if isinstance(img, str):
        if img.startswith('http'):
            img = requests.get(img, stream=True).raw
            img = Image.open(img)
        else:
            img = Path(img)
    if isinstance(img, Path):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        img = np.array(img)
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if not isinstance(img, np.ndarray):
        raise TypeError(f'Unsupported input type: {type(img)}')
    if len(img.shape) != 3:
        raise ValueError(f'Unsupported input shape: {img.shape}')
    if img.shape[2] not in [1, 3]:
        raise ValueError(f'Unsupported input shape: {img.shape}')
    return img
