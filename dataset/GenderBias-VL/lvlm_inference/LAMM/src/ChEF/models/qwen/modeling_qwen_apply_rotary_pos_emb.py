def apply_rotary_pos_emb(t, freqs):
    cos, sin = freqs
    if apply_rotary_emb_func is not None and t.is_cuda:
        t_ = t.float()
        cos = cos.squeeze(0).squeeze(1)[:, :cos.shape[-1] // 2]
        sin = sin.squeeze(0).squeeze(1)[:, :sin.shape[-1] // 2]
        output = apply_rotary_emb_func(t_, cos, sin).type_as(t)
        return output
    else:
        rot_dim = freqs[0].shape[-1]
        cos, sin = freqs
        t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
        t_ = t_.float()
        t_pass_ = t_pass_.float()
        t_ = t_ * cos + _rotate_half(t_) * sin
        return torch.cat((t_, t_pass_), dim=-1).type_as(t)
