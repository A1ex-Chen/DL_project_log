def binarize(self, image: PIL.Image.Image) ->PIL.Image.Image:
    """
        Create a mask.

        Args:
            image (`PIL.Image.Image`):
                The image input, should be a PIL image.

        Returns:
            `PIL.Image.Image`:
                The binarized image. Values less than 0.5 are set to 0, values greater than 0.5 are set to 1.
        """
    image[image < 0.5] = 0
    image[image >= 0.5] = 1
    return image
