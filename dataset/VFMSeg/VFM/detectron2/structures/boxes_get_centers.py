def get_centers(self) ->torch.Tensor:
    """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
    return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2
