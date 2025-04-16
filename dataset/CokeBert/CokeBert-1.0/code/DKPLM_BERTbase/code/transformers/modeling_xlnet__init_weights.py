def _init_weights(self, module):
    """ Initialize the weights.
        """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, XLNetLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, XLNetRelativeAttention):
        for param in [module.q, module.k, module.v, module.o, module.r,
            module.r_r_bias, module.r_s_bias, module.r_w_bias, module.seg_embed
            ]:
            param.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, XLNetModel):
        module.mask_emb.data.normal_(mean=0.0, std=self.config.
            initializer_range)
