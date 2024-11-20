def post_attention_pooling(self, attention_inputs):
    """ Pool the proper parts of `attention_inputs` after the attention layer. """
    position_embeds, token_type_mat, attention_mask, cls_mask = (
        attention_inputs)
    if self.pool_q_only:
        self.pooling_mult *= 2
        if self.attention_type == 'factorized':
            position_embeds = position_embeds[:2] + self.stride_pool(
                position_embeds[2:], 0)
        token_type_mat = self.stride_pool(token_type_mat, 2)
        cls_mask = self.stride_pool(cls_mask, 1)
        attention_mask = self.pool_tensor(attention_mask, mode='min')
    attention_inputs = (position_embeds, token_type_mat, attention_mask,
        cls_mask)
    return attention_inputs
