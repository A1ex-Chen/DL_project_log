def encode(self, x, x_lens):
    """
        Args:
            x: tuple of ``(input, input_lens)``. ``input`` has shape (T, B, I),
                ``input_lens`` has shape ``(B,)``.

        Returns:
            f: tuple of ``(output, output_lens)``. ``output`` has shape
                (B, T, H), ``output_lens``
        """
    require_padding = type(x) is not list and x.size(1) < self.min_lstm_bs
    if require_padding:
        bs = x.size(1)
        x = torch.nn.functional.pad(x, (0, 0, 0, self.min_lstm_bs - bs))
    x, _ = self.encoder['pre_rnn'](x, None)
    x, x_lens = self.encoder['stack_time'](x, x_lens)
    x, _ = self.encoder['post_rnn'](x, None)
    if type(x) is list:
        x = self._seq_merge(x)
    if require_padding:
        x = x[:, :bs]
    f = self.joint_enc(x.transpose(0, 1))
    return f, x_lens
