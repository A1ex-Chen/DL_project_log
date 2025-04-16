def new_like(self, new_input_size=None):
    """
        new_like()
        """
    if new_input_size is None:
        new_input_size = self.input_size
    return type(self)(self.gate_multiplier, new_input_size, self.
        hidden_size, self.cell, self.n_hidden_states, self.bias, self.
        output_size)
