def encode_pcl(self, pcl_paths):
    inputs = self.load_and_transform_pcl_data(pcl_paths, self.device)
    inputs = inputs.to(self.llama_model.dtype)
    with torch.no_grad():
        if self.vision_feature_type == 'global':
            raise NotImplementedError('Global feature not implemented for pcl')
        elif self.vision_feature_type == 'local':
            embeddings = self.visual_encoder(inputs)[1][:, :self.
                num_vision_token]
            image_embeds = embeddings.reshape(-1, self.vision_hidden_size).to(
                self.llama_model.dtype)
    inputs_llama = self.llama_proj(image_embeds).reshape(-1, self.
        num_vision_token, self.llama_model.config.hidden_size)
    atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self
        .device)
    return inputs_llama, atts_llama
