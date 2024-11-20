@property
def dummy_image_encoder(self):
    torch.manual_seed(0)
    config = CLIPVisionConfig(hidden_size=self.text_embedder_hidden_size,
        image_size=32, projection_dim=self.text_embedder_hidden_size,
        intermediate_size=24, num_attention_heads=2, num_channels=3,
        num_hidden_layers=5, patch_size=1)
    model = CLIPVisionModel(config)
    return model
