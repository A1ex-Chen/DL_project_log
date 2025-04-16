def apply_rotary_pos_emb(t, freqs):
    """ Apply rotary embedding to the first rotary_dim of the iput
    Arguments:
      t (tensor(batch_size, seq_len, n_head, head_dim)):
        the input embedding/hidden states
      freqs (list[tensor(1, seq_len, 1, rotary_dim), tensor(1, seq_len, 1, rotary_dim)]):
        the cached cos/sin position embeddings
    """
    rot_dim = freqs[0].shape[-1]
    cos, sin = freqs
    t_float = t.float()
    if apply_rotary_emb_func is not None and t.is_cuda:
        cos = cos.squeeze(0).squeeze(1)[:, :rot_dim // 2]
        sin = sin.squeeze(0).squeeze(1)[:, :rot_dim // 2]
        return apply_rotary_emb_func(t_float, cos, sin).type_as(t)
    else:
        t_rot, t_pass = t_float[..., :rot_dim], t_float[..., rot_dim:]
        t_rot = t_rot * cos + _rotate_half(t_rot) * sin
        return torch.cat((t_rot, t_pass), dim=-1).type_as(t)
