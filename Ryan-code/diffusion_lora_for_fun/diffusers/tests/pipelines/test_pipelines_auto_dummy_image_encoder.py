@property
def dummy_image_encoder(self):
    torch.manual_seed(0)
    config = CLIPVisionConfig(hidden_size=1, projection_dim=1,
        num_hidden_layers=1, num_attention_heads=1, image_size=1,
        intermediate_size=1, patch_size=1)
    return CLIPVisionModelWithProjection(config)
