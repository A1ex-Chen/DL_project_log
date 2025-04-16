def init_attention_inputs(self, inputs_embeds, attention_mask=None,
    token_type_ids=None):
    """ Returns the attention inputs associated to the inputs of the model. """
    self.pooling_mult = 1
    self.seq_len = seq_len = inputs_embeds.size(1)
    position_embeds = self.get_position_embeds(seq_len, inputs_embeds.dtype,
        inputs_embeds.device)
    token_type_mat = self.token_type_ids_to_mat(token_type_ids
        ) if token_type_ids is not None else None
    cls_mask = F.pad(inputs_embeds.new_ones([seq_len - 1, seq_len - 1]), (1,
        0, 1, 0)) if self.config.separate_cls else None
    return position_embeds, token_type_mat, attention_mask, cls_mask
