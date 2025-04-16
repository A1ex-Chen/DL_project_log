@staticmethod
def convert_to_rgb(image: PIL.Image.Image) ->PIL.Image.Image:
    """
        Converts a PIL image to RGB format.
        """
    image = image.convert('RGB')
    return image
