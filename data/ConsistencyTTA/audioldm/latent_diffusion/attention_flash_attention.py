def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
        #### Flash Attention
        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """
    batch_size, seq_len, _ = q.shape
    qkv = torch.stack((q, k, v), dim=2)
    qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head)
    if self.d_head <= 32:
        pad = 32 - self.d_head
    elif self.d_head <= 64:
        pad = 64 - self.d_head
    elif self.d_head <= 128:
        pad = 128 - self.d_head
    else:
        raise ValueError(
            f'Head size ${self.d_head} too large for Flash Attention')
    if pad:
        qkv = torch.cat((qkv, qkv.new_zeros(batch_size, seq_len, 3, self.
            n_heads, pad)), dim=-1)
    out, _ = self.flash(qkv.type(torch.float16))
    out = out[:, :, :, :self.d_head].float()
    out = out.reshape(batch_size, seq_len, self.n_heads * self.d_head)
    return self.to_out(out)
