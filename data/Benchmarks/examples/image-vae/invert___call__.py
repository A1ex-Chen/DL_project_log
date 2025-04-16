def __call__(self, img):
    """
        Args:
            img (PIL Image): Image to be inverted.
        Returns:
            PIL Image: Inverted image.
        """
    return self.invert(img)
