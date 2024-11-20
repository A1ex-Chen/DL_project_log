def forward(self, attn, is_cross: bool, place_in_unet: str):
    super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
    if is_cross or self.num_self_replace[0
        ] <= self.cur_step < self.num_self_replace[1]:
        h = attn.shape[0] // self.batch_size
        attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
        attn_base, attn_repalce = attn[0], attn[1:]
        if is_cross:
            alpha_words = self.cross_replace_alpha[self.cur_step]
            attn_repalce_new = self.replace_cross_attention(attn_base,
                attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
            attn[1:] = attn_repalce_new
        else:
            attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
        attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
    return attn
