def decode(self, inputs, context, inference=False):
    """
        Applies the decoder to inputs, given the context from the encoder.

        :param inputs: tensor with inputs (batch, seq_len) if 'batch_first'
            else (seq_len, batch)
        :param context: context from the encoder
        :param inference: if True inference mode, if False training mode
        """
    return self.decoder(inputs, context, inference)
