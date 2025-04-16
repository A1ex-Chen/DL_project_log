def new_like(self, new_input_size=None):
    if new_input_size is None:
        new_input_size = self.input_size
    return type(self)(new_input_size, self.hidden_size, self.bias, self.
        output_size)
