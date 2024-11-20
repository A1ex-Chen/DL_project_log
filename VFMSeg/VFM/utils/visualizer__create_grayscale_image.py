def _create_grayscale_image(self, mask=None):
    """
        Create a grayscale version of the original image.
        The colors in masked area, if given, will be kept.
        """
    img_bw = self.img.astype('f4').mean(axis=2)
    img_bw = np.stack([img_bw] * 3, axis=2)
    if mask is not None:
        img_bw[mask] = self.img[mask]
    return img_bw
