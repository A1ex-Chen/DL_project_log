def forward(self, x, **kwargs):
    x_inp, x_labels = x[:, :-1], x[:, 1:]
    logits = self.net(x_inp, **kwargs)
    return F.cross_entropy(rearrange(logits, 'b c n -> b n c'), x_labels)
