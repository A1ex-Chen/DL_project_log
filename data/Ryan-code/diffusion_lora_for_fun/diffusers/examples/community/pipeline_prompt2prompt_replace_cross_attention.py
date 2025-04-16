def replace_cross_attention(self, attn_base, att_replace):
    if self.prev_controller is not None:
        attn_base = self.prev_controller.replace_cross_attention(attn_base,
            att_replace)
    attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
    return attn_replace
