def package_hidden(self):
    """
        Flattens the hidden state from all LSTM layers into one tensor (for
        the sequence generator).
        """
    if self.inference:
        hidden = torch.cat(tuple(itertools.chain(*self.next_hidden)))
    else:
        hidden = None
    return hidden
