@staticmethod
def numpy_to_pt(images):
    """
        Convert a numpy image to a pytorch tensor
        """
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images
