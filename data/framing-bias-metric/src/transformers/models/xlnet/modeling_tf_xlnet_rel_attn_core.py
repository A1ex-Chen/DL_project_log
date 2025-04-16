def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat,
    attn_mask, head_mask, output_attentions, training=False):
    """Core relative positional attention operations."""
    ac = tf.einsum('ibnd,jbnd->ijbn', q_head + self.r_w_bias, k_head_h)
    bd = tf.einsum('ibnd,jbnd->ijbn', q_head + self.r_r_bias, k_head_r)
    bd = self.rel_shift(bd, klen=shape_list(ac)[1])
    if seg_mat is None:
        ef = 0
    else:
        ef = tf.einsum('ibnd,snd->ibns', q_head + self.r_s_bias, self.seg_embed
            )
        ef = tf.einsum('ijbs,ibns->ijbn', seg_mat, ef)
    attn_score = (ac + bd + ef) * self.scale
    if attn_mask is not None:
        if attn_mask.dtype == tf.float16:
            attn_score = attn_score - 65500 * attn_mask
        else:
            attn_score = attn_score - 1e+30 * attn_mask
    attn_prob = tf.nn.softmax(attn_score, axis=1)
    attn_prob = self.dropout(attn_prob, training=training)
    if head_mask is not None:
        attn_prob = attn_prob * head_mask
    attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)
    if output_attentions:
        return attn_vec, attn_prob
    return attn_vec
