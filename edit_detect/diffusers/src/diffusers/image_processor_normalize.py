@staticmethod
def normalize(images: Union[np.ndarray, torch.Tensor]) ->Union[np.ndarray,
    torch.Tensor]:
    """
        Normalize an image array to [-1,1].
        """
    return 2.0 * images - 1.0
