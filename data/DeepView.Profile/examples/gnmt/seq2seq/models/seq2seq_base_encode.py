def encode(self, inputs, lengths):
    """
        Applies the encoder to inputs with a given input sequence lengths.

        :param inputs: tensor with inputs (batch, seq_len) if 'batch_first'
            else (seq_len, batch)
        :param lengths: vector with sequence lengths (excluding padding)
        """
    return self.encoder(inputs, lengths)
