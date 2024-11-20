def init_hidden(self, hidden):
    """
        Converts flattened hidden state (from sequence generator) into a tuple
        of hidden states.

        :param hidden: None or flattened hidden state for decoder RNN layers
        """
    if hidden is not None:
        hidden = hidden.chunk(self.num_layers)
        hidden = tuple(i.chunk(2) for i in hidden)
    else:
        hidden = [None] * self.num_layers
    self.next_hidden = []
    return hidden
