def _init_weights(self, module):
    """Initialize the weights"""
    factor = self.config.initializer_factor
    if isinstance(module, T5LayerNorm):
        module.weight.data.fill_(factor * 1.0)
    elif isinstance(module, (T5Model, T5ForConditionalGeneration,
        T5EncoderModel)):
        module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        if hasattr(module, 'lm_head') and not self.config.tie_word_embeddings:
            module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
    elif isinstance(module, T5DenseActDense):
        module.wi.weight.data.normal_(mean=0.0, std=factor * self.config.
            d_model ** -0.5)
        if hasattr(module.wi, 'bias') and module.wi.bias is not None:
            module.wi.bias.data.zero_()
        module.wo.weight.data.normal_(mean=0.0, std=factor * self.config.
            d_ff ** -0.5)
        if hasattr(module.wo, 'bias') and module.wo.bias is not None:
            module.wo.bias.data.zero_()
    elif isinstance(module, T5DenseGatedActDense):
        module.wi_0.weight.data.normal_(mean=0.0, std=factor * self.config.
            d_model ** -0.5)
        if hasattr(module.wi_0, 'bias') and module.wi_0.bias is not None:
            module.wi_0.bias.data.zero_()
        module.wi_1.weight.data.normal_(mean=0.0, std=factor * self.config.
            d_model ** -0.5)
        if hasattr(module.wi_1, 'bias') and module.wi_1.bias is not None:
            module.wi_1.bias.data.zero_()
        module.wo.weight.data.normal_(mean=0.0, std=factor * self.config.
            d_ff ** -0.5)
        if hasattr(module.wo, 'bias') and module.wo.bias is not None:
            module.wo.bias.data.zero_()
    elif isinstance(module, T5Attention):
        d_model = self.config.d_model
        key_value_proj_dim = self.config.d_kv
        n_heads = self.config.num_heads
        module.q.weight.data.normal_(mean=0.0, std=factor * (d_model *
            key_value_proj_dim) ** -0.5)
        module.k.weight.data.normal_(mean=0.0, std=factor * d_model ** -0.5)
        module.v.weight.data.normal_(mean=0.0, std=factor * d_model ** -0.5)
        module.o.weight.data.normal_(mean=0.0, std=factor * (n_heads *
            key_value_proj_dim) ** -0.5)
        if module.has_relative_attention_bias:
            module.relative_attention_bias.weight.data.normal_(mean=0.0,
                std=factor * d_model ** -0.5)
