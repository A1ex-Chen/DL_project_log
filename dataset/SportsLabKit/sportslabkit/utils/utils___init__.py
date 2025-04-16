def __init__(self, path: str):
    """Very simple iterator class for image files.

        Args:
            path (str): Path to image file
        """
    assert os.path.isdir(path), f'{path} is not a directory.'
    self.path = path
    imgs = []
    valid_images = ['.jpg', '.gif', '.png', '.tga']
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(cv.imread(os.path.join(path, f)))
    self.imgs = imgs
    self._index = 0
