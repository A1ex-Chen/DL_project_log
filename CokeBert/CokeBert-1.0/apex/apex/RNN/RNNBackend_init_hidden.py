def init_hidden(self, bsz):
    """
        init_hidden()
        """
    for param in self.parameters():
        if param is not None:
            a_param = param
            break
    for i, _ in enumerate(self.hidden):
        if self.hidden[i] is None or self.hidden[i].data.size()[0] != bsz:
            if i == 0:
                hidden_size = self.output_size
            else:
                hidden_size = self.hidden_size
            tens = a_param.data.new(bsz, hidden_size).zero_()
            self.hidden[i] = Variable(tens, requires_grad=False)
