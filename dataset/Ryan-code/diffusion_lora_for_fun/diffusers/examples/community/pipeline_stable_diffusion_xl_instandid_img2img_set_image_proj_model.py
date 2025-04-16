def set_image_proj_model(self, model_ckpt, image_emb_dim=512, num_tokens=16):
    image_proj_model = Resampler(dim=1280, depth=4, dim_head=64, heads=20,
        num_queries=num_tokens, embedding_dim=image_emb_dim, output_dim=
        self.unet.config.cross_attention_dim, ff_mult=4)
    image_proj_model.eval()
    self.image_proj_model = image_proj_model.to(self.device, dtype=self.dtype)
    state_dict = torch.load(model_ckpt, map_location='cpu')
    if 'image_proj' in state_dict:
        state_dict = state_dict['image_proj']
    self.image_proj_model.load_state_dict(state_dict)
    self.image_proj_model_in_features = image_emb_dim
