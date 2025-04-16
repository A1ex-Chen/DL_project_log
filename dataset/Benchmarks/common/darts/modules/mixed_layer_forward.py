def forward(self, x, weights):
    """
        Parameters
        ----------
        x : torch.tensor
            Data

        Weights : torch.tensor
            alpha, [op_num:8], the output = sum of alpha * op(x)
        """
    x = [(w * layer(x)) for w, layer in zip(weights, self.layers)]
    x = self.pad(x)
    return sum(x)
