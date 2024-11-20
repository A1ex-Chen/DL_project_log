@staticmethod
def blur(image: PIL.Image.Image, blur_factor: int=4) ->PIL.Image.Image:
    """
        Applies Gaussian blur to an image.
        """
    image = image.filter(ImageFilter.GaussianBlur(blur_factor))
    return image
