def _convert_to_dual_attention(self):
    """
        Replace image_unet's `Transformer2DModel` blocks with `DualTransformer2DModel` that contains transformer blocks
        from both `image_unet` and `text_unet`
        """
    for name, module in self.image_unet.named_modules():
        if isinstance(module, Transformer2DModel):
            parent_name, index = name.rsplit('.', 1)
            index = int(index)
            image_transformer = self.image_unet.get_submodule(parent_name)[
                index]
            text_transformer = self.text_unet.get_submodule(parent_name)[index]
            config = image_transformer.config
            dual_transformer = DualTransformer2DModel(num_attention_heads=
                config.num_attention_heads, attention_head_dim=config.
                attention_head_dim, in_channels=config.in_channels,
                num_layers=config.num_layers, dropout=config.dropout,
                norm_num_groups=config.norm_num_groups, cross_attention_dim
                =config.cross_attention_dim, attention_bias=config.
                attention_bias, sample_size=config.sample_size,
                num_vector_embeds=config.num_vector_embeds, activation_fn=
                config.activation_fn, num_embeds_ada_norm=config.
                num_embeds_ada_norm)
            dual_transformer.transformers[0] = image_transformer
            dual_transformer.transformers[1] = text_transformer
            self.image_unet.get_submodule(parent_name)[index
                ] = dual_transformer
            self.image_unet.register_to_config(dual_cross_attention=True)
