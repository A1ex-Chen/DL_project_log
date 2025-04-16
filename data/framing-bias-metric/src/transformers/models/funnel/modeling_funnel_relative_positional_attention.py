def relative_positional_attention(self, position_embeds, q_head,
    context_len, cls_mask=None):
    """ Relative attention score for the positional encodings """
    if self.config.attention_type == 'factorized':
        phi, pi, psi, omega = position_embeds
        u = self.r_r_bias * self.scale
        w_r = self.r_kernel
        q_r_attention = torch.einsum('binh,dnh->bind', q_head + u, w_r)
        q_r_attention_1 = q_r_attention * phi[:, None]
        q_r_attention_2 = q_r_attention * pi[:, None]
        positional_attn = torch.einsum('bind,jd->bnij', q_r_attention_1, psi
            ) + torch.einsum('bind,jd->bnij', q_r_attention_2, omega)
    else:
        shift = 2 if q_head.shape[1] != context_len else 1
        r = position_embeds[self.block_index][shift - 1]
        v = self.r_r_bias * self.scale
        w_r = self.r_kernel
        r_head = torch.einsum('td,dnh->tnh', r, w_r)
        positional_attn = torch.einsum('binh,tnh->bnit', q_head + v, r_head)
        positional_attn = _relative_shift_gather(positional_attn,
            context_len, shift)
    if cls_mask is not None:
        positional_attn *= cls_mask
    return positional_attn
