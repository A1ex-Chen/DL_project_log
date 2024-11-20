def from_numpy(self, x):
    """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
    return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x
