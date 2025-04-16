def _mask_pads(ll, smooth_obj):
    pad_mask = target.eq(self.config.generator.pad_token_id)
    if pad_mask.any():
        ll.masked_fill_(pad_mask, 0.0)
        smooth_obj.masked_fill_(pad_mask, 0.0)
    return ll.squeeze(-1), smooth_obj.squeeze(-1)
