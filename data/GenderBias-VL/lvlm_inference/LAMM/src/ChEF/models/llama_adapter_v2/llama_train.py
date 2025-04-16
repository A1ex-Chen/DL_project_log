def train(self, mode: bool=True):
    if mode:
        self.cache_k = None
        self.cache_v = None
    else:
        self.cache_k = torch.zeros((self.args.max_batch_size, self.args.
            max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.cache_v = torch.zeros((self.args.max_batch_size, self.args.
            max_seq_len, self.n_local_heads, self.head_dim)).cuda()
    return super().train(mode)
