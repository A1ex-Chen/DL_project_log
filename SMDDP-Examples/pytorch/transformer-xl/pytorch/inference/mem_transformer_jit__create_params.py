def _create_params(self):
    if self.attn_type == 0:
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head)
            .zero_())
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head)
            .zero_())
