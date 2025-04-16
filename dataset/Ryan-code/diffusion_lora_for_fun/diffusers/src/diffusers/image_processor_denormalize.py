@staticmethod
def denormalize(images: Union[np.ndarray, torch.Tensor]) ->Union[np.ndarray,
    torch.Tensor]:
    """
        Denormalize an image array to [0,1].
        """
    return (images / 2 + 0.5).clamp(0, 1)
