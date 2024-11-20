@staticmethod
def pt_to_numpy(images: torch.Tensor) ->np.ndarray:
    """
        Convert a PyTorch tensor to a NumPy image.
        """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images
