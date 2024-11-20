def forward(self, inputs, context, inference=False):
    """
        Execute the decoder.

        :param inputs: tensor with inputs to the decoder
        :param context: state of encoder, encoder sequence lengths and hidden
            state of decoder's LSTM layers
        :param inference: if True stores and repackages hidden state
        """
    self.inference = inference
    enc_context, enc_len, hidden = context
    hidden = self.init_hidden(hidden)
    x = self.embedder(inputs)
    x, h, attn, scores = self.att_rnn(x, hidden[0], enc_context, enc_len)
    self.append_hidden(h)
    x = torch.cat((x, attn), dim=2)
    x = self.dropout(x)
    x, h = self.rnn_layers[0](x, hidden[1])
    self.append_hidden(h)
    for i in range(1, len(self.rnn_layers)):
        residual = x
        x = torch.cat((x, attn), dim=2)
        x = self.dropout(x)
        x, h = self.rnn_layers[i](x, hidden[i + 1])
        self.append_hidden(h)
        x = x + residual
    x = self.classifier(x)
    hidden = self.package_hidden()
    return x, scores, [enc_context, enc_len, hidden]
