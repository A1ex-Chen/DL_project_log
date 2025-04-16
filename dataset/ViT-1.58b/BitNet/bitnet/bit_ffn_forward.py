def forward(self, x):
    """
        Forward pass of the BitFeedForward module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
    return self.ff(x)
