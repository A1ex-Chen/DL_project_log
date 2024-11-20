def forward(self, x: Tensor) ->Tensor:
    """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
    b, s, d = x.shape
    w = self.weight
    x_norm = RMSNorm(d)(x)
    x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
    w_quant = w + (weight_quant(w) - w).detach()
    y = F.linear(x_quant, w_quant)
    return y
