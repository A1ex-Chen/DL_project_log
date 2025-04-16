def init_weights(self):
    for param in self.vision_encoder.parameters():
        param.requires_grad = False
    for name, param in self.lang_encoder.named_parameters():
        if 'gated_cross_attn_layer' not in name:
            param.requires_grad = False
    self.lang_encoder.get_input_embeddings().requires_grad_(True)
    self.lang_encoder.get_output_embeddings().requires_grad_(True)
