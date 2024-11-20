def __init__(self, encoder=None, decoder=None, batch_first=False):
    """
        Constructor for the Seq2Seq module.

        :param encoder: encoder module
        :param decoder: decoder module
        :param batch_first: if True the model uses (batch, seq, feature)
            tensors, if false the model uses (seq, batch, feature) tensors
        """
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.batch_first = batch_first
