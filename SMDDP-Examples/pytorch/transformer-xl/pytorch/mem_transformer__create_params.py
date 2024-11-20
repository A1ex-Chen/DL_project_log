def _create_params(self):
    if self.attn_type == 0:
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head)
            .zero_())
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head)
            .zero_())
    elif self.attn_type == 1:
        self.r_emb = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen,
            self.n_head, self.d_head).zero_())
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_layer, self.n_head,
            self.d_head).zero_())
        self.r_bias = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen,
            self.n_head).zero_())
    elif self.attn_type == 2:
        self.pos_emb = PositionalEmbedding(self.d_model)
    elif self.attn_type == 3:
        self.r_emb = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen,
            self.d_model).zero_())
