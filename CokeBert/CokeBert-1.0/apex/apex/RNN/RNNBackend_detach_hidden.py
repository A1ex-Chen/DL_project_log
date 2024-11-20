def detach_hidden(self):
    """
        detach_hidden()
        """
    for i, _ in enumerate(self.hidden):
        if self.hidden[i] is None:
            raise RuntimeError(
                'Must initialize hidden state before you can detach it')
    for i, _ in enumerate(self.hidden):
        self.hidden[i] = self.hidden[i].detach()
