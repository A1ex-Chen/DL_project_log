def append_hidden(self, h):
    """
        Appends the hidden vector h to the list of internal hidden states.

        :param h: hidden vector
        """
    if self.inference:
        self.next_hidden.append(h)
