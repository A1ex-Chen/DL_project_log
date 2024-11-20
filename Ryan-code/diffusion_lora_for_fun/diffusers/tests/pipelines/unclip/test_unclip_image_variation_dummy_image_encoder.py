@property
def dummy_image_encoder(self):
    torch.manual_seed(0)
    config = CLIPVisionConfig(hidden_size=self.text_embedder_hidden_size,
        projection_dim=self.text_embedder_hidden_size, num_hidden_layers=5,
        num_attention_heads=4, image_size=32, intermediate_size=37,
        patch_size=1)
    return CLIPVisionModelWithProjection(config)
