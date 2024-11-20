@staticmethod
def numpy_to_pt(images: np.ndarray) ->torch.Tensor:
    """
        Convert a NumPy image to a PyTorch tensor.
        """
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images
