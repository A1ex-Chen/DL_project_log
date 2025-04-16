def _norm(self, x):
    """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
