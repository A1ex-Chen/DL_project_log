def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos.unsqueeze(0).unsqueeze(0).expand(len(position_ids), -1, -1, -1)
    sin = sin.unsqueeze(0).unsqueeze(0).expand(len(position_ids), -1, -1, -1)
    if q.size(2) == 1:
        q_embed = q * cos[:, :, -1:, :] + rotate_half(q) * sin[:, :, -1:, :]
    else:
        q_embed = q * cos + rotate_half(q) * sin
    if k.size(2) == 1:
        k_embed = k * cos[:, :, -1:, :] + rotate_half(k) * sin[:, :, -1:, :]
    else:
        k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed
