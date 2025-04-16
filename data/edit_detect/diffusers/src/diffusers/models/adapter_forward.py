def forward(self, x: torch.Tensor) ->torch.Tensor:
    """
        This function takes input tensor x and processes it through one convolutional layer, ReLU activation, and
        another convolutional layer and adds it to input tensor.
        """
    h = self.act(self.block1(x))
    h = self.block2(h)
    return h + x
