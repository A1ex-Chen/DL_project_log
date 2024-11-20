def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
    num_layers: int, sigmoid_output: bool=False) ->None:
    """
        Initializes the MLP (Multi-Layer Perceptron) model.

        Args:
            input_dim (int): The dimensionality of the input features.
            hidden_dim (int): The dimensionality of the hidden layers.
            output_dim (int): The dimensionality of the output layer.
            num_layers (int): The number of hidden layers.
            sigmoid_output (bool, optional): Apply a sigmoid activation to the output layer. Defaults to False.
        """
    super().__init__()
    self.num_layers = num_layers
    h = [hidden_dim] * (num_layers - 1)
    self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] +
        h, h + [output_dim]))
    self.sigmoid_output = sigmoid_output
