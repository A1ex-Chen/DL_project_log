def set_transformer_params(self, mix_ratio: float=0.5, condition_types:
    Tuple=('text', 'image')):
    for name, module in self.image_unet.named_modules():
        if isinstance(module, DualTransformer2DModel):
            module.mix_ratio = mix_ratio
            for i, type in enumerate(condition_types):
                if type == 'text':
                    module.condition_lengths[i
                        ] = self.text_encoder.config.max_position_embeddings
                    module.transformer_index_for_condition[i] = 1
                else:
                    module.condition_lengths[i] = 257
                    module.transformer_index_for_condition[i] = 0
