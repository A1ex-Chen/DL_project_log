@property
def dummy_image_encoder(self):
    torch.manual_seed(0)
    config = CLIPVisionConfig(hidden_size=self.text_embedder_hidden_size,
        image_size=224, projection_dim=self.text_embedder_hidden_size,
        intermediate_size=37, num_attention_heads=4, num_channels=3,
        num_hidden_layers=5, patch_size=14)
    model = CLIPVisionModelWithProjection(config)
    return model
