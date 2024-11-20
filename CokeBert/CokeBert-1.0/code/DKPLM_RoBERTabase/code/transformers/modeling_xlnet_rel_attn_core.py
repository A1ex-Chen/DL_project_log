def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat=None,
    attn_mask=None, head_mask=None):
    """Core relative positional attention operations."""
    ac = torch.einsum('ibnd,jbnd->bnij', q_head + self.r_w_bias, k_head_h)
    bd = torch.einsum('ibnd,jbnd->bnij', q_head + self.r_r_bias, k_head_r)
    bd = self.rel_shift_bnij(bd, klen=ac.shape[3])
    if seg_mat is None:
        ef = 0
    else:
        ef = torch.einsum('ibnd,snd->ibns', q_head + self.r_s_bias, self.
            seg_embed)
        ef = torch.einsum('ijbs,ibns->bnij', seg_mat, ef)
    attn_score = (ac + bd + ef) * self.scale
    if attn_mask is not None:
        if attn_mask.dtype == torch.float16:
            attn_score = attn_score - 65500 * torch.einsum('ijbn->bnij',
                attn_mask)
        else:
            attn_score = attn_score - 1e+30 * torch.einsum('ijbn->bnij',
                attn_mask)
    attn_prob = F.softmax(attn_score, dim=3)
    attn_prob = self.dropout(attn_prob)
    if head_mask is not None:
        attn_prob = attn_prob * torch.einsum('ijbn->bnij', head_mask)
    attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, v_head_h)
    if self.output_attentions:
        return attn_vec, torch.einsum('bnij->ijbn', attn_prob)
    return attn_vec
