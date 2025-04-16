def forward(self, x, kv):
    kv = self.kv_mapper(kv)
    norm_x = self.norm(x)
    if self.self_attn:
        batch_size, channel, _, _ = x.shape
        kv = torch.cat([norm_x.view(batch_size, channel, -1).transpose(1, 2
            ), kv], dim=1)
    x = x + self.attention(norm_x, encoder_hidden_states=kv)
    return x
