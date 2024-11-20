@staticmethod
def convert_to_grayscale(image: PIL.Image.Image) ->PIL.Image.Image:
    """
        Converts a PIL image to grayscale format.
        """
    image = image.convert('L')
    return image
