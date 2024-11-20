def compute_ca_loss(self, saved_attn, bboxes, phrase_indices,
    guidance_attn_keys, verbose=False, **kwargs):
    """
        The `saved_attn` is supposed to be passed to `save_attn_to_dict` in `cross_attention_kwargs` prior to computing ths loss.
        `AttnProcessor` will put attention maps into the `save_attn_to_dict`.

        `index` is the timestep.
        `ref_ca_word_token_only`: This has precedence over `ref_ca_last_token_only` (i.e., if both are enabled, we take the token from word rather than the last token).
        `ref_ca_last_token_only`: `ref_ca_saved_attn` comes from the attention map of the last token of the phrase in single object generation, so we apply it only to the last token of the phrase in overall generation if this is set to True. If set to False, `ref_ca_saved_attn` will be applied to all the text tokens.
        """
    loss = torch.tensor(0).float().cuda()
    object_number = len(bboxes)
    if object_number == 0:
        return loss
    for attn_key in guidance_attn_keys:
        attn_map_integrated = saved_attn[attn_key]
        if not attn_map_integrated.is_cuda:
            attn_map_integrated = attn_map_integrated.cuda()
        attn_map = attn_map_integrated.squeeze(dim=0)
        loss = self.add_ca_loss_per_attn_map_to_loss(loss, attn_map,
            object_number, bboxes, phrase_indices, **kwargs)
    num_attn = len(guidance_attn_keys)
    if num_attn > 0:
        loss = loss / (object_number * num_attn)
    return loss
