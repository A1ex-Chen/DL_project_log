def relative_token_type_attention(self, token_type_mat, q_head, cls_mask=None):
    """ Relative attention score for the token_type_ids """
    if token_type_mat is None:
        return 0
    batch_size, seq_len, context_len = token_type_mat.shape
    r_s_bias = self.r_s_bias * self.scale
    token_type_bias = torch.einsum('bind,snd->bnis', q_head + r_s_bias,
        self.seg_embed)
    token_type_mat = token_type_mat[:, None].expand([batch_size, q_head.
        shape[2], seq_len, context_len])
    diff_token_type, same_token_type = torch.split(token_type_bias, 1, dim=-1)
    token_type_attn = torch.where(token_type_mat, same_token_type.expand(
        token_type_mat.shape), diff_token_type.expand(token_type_mat.shape))
    if cls_mask is not None:
        token_type_attn *= cls_mask
    return token_type_attn
