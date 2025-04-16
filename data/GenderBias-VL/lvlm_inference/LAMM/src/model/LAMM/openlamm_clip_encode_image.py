def clip_encode_image(self, inputs):
    inputs = inputs.to(dtype=self.llama_model.dtype)
    if self.vision_feature_type == 'global':
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
        image_embeds = embeddings.to(self.llama_model.dtype)
        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1)
    elif self.vision_feature_type == 'local':
        with torch.no_grad():
            embeddings = self.visual_encoder.forward_patch_features(inputs.
                to(self.device))[:, :self.num_vision_token]
        image_embeds = embeddings.reshape(-1, self.vision_hidden_size).to(self
            .llama_model.dtype)
        inputs_llama = self.llama_proj(image_embeds).reshape(-1, self.
            num_vision_token, self.llama_model.config.hidden_size)
    else:
        raise NotImplementedError('{} not Implemented'.format(self.
            vision_feature_type))
    return inputs_llama
