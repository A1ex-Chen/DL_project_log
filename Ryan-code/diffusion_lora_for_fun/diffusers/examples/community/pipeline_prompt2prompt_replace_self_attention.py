def replace_self_attention(self, attn_base, att_replace):
    if att_replace.shape[2] <= 16 ** 2:
        return attn_base.unsqueeze(0).expand(att_replace.shape[0], *
            attn_base.shape)
    else:
        return att_replace
