def encode_image(self, images):
    with torch.no_grad():
        vision_embeds_2d = self.visual_encoder.forward_patch_features(images)[
            :, :self.num_vision_token]
    vision_embeds_2d = vision_embeds_2d.reshape(-1, self.vision_hidden_size
        ).to(self.llama_model.dtype)
    vision_embeds_2d = self.llama_proj(vision_embeds_2d).reshape(-1, self.
        num_vision_token, self.llama_model.config.hidden_size)
    return vision_embeds_2d
