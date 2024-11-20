def forward(self, inputs, lengths):
    """
        Execute the encoder.

        :param inputs: tensor with indices from the vocabulary
        :param lengths: vector with sequence lengths (excluding padding)

        returns: tensor with encoded sequences
        """
    x = self.embedder(inputs)
    x = self.dropout(x)
    x = pack_padded_sequence(x, lengths.cpu().numpy(), batch_first=self.
        batch_first)
    x, _ = self.rnn_layers[0](x)
    x, _ = pad_packed_sequence(x, batch_first=self.batch_first)
    x = self.dropout(x)
    x, _ = self.rnn_layers[1](x)
    for i in range(2, len(self.rnn_layers)):
        residual = x
        x = self.dropout(x)
        x, _ = self.rnn_layers[i](x)
        x = x + residual
    return x
