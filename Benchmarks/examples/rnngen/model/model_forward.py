def forward(self, x, with_softmax=False):
    x = [self.emb(x_) for x_ in x]
    x = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
    x, _ = self.lstm(x)
    x, _ = nn.utils.rnn.pad_packed_sequence(x, padding_value=0,
        total_length=self.max_len)
    x = self.linear(x)
    if with_softmax:
        return F.softmax(x, dim=-1)
    else:
        return x
