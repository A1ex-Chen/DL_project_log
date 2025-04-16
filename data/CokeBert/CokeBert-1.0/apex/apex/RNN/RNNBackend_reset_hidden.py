def reset_hidden(self, bsz):
    """
        reset_hidden()
        """
    for i, _ in enumerate(self.hidden):
        self.hidden[i] = None
    self.init_hidden(bsz)
