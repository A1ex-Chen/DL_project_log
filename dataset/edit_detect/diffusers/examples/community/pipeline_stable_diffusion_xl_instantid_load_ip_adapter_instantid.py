def load_ip_adapter_instantid(self, model_ckpt, image_emb_dim=512,
    num_tokens=16, scale=0.5):
    self.set_image_proj_model(model_ckpt, image_emb_dim, num_tokens)
    self.set_ip_adapter(model_ckpt, num_tokens, scale)
