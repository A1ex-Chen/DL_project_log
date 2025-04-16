def forward(self, x):
    """
        Forward pass of the SCDown module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the SCDown module.
        """
    return self.cv2(self.cv1(x))
