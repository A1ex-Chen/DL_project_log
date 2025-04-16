def forward(self, te: torch.Tensor):
    te_tokens = self.embedding(te)
    past_key_values = self.trans(te_tokens)
    return past_key_values
