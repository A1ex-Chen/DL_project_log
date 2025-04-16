@staticmethod
def normalize(images):
    """
        Normalize an image array to [-1,1]
        """
    return 2.0 * images - 1.0
