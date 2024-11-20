def pre_attention_pooling(self, output, attention_inputs):
    """ Pool `output` and the proper parts of `attention_inputs` before the attention layer. """
    position_embeds, token_type_mat, attention_mask, cls_mask = (
        attention_inputs)
    if self.config.pool_q_only:
        if self.config.attention_type == 'factorized':
            position_embeds = self.stride_pool(position_embeds[:2], 0
                ) + position_embeds[2:]
        token_type_mat = self.stride_pool(token_type_mat, 1)
        cls_mask = self.stride_pool(cls_mask, 0)
        output = self.pool_tensor(output, mode=self.config.pooling_type)
    else:
        self.pooling_mult *= 2
        if self.config.attention_type == 'factorized':
            position_embeds = self.stride_pool(position_embeds, 0)
        token_type_mat = self.stride_pool(token_type_mat, [1, 2])
        cls_mask = self.stride_pool(cls_mask, [1, 2])
        attention_mask = self.pool_tensor(attention_mask, mode='min')
        output = self.pool_tensor(output, mode=self.config.pooling_type)
    attention_inputs = (position_embeds, token_type_mat, attention_mask,
        cls_mask)
    return output, attention_inputs
