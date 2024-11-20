def forward(self, input):
    """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
    return self.conv(input, weight=self.weight.to(input.dtype), groups=self
        .groups)
