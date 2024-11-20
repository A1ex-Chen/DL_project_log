def _get_dummy_faceid_image_embeds(self, cross_attention_dim: int=32):
    return torch.randn((2, 1, 1, cross_attention_dim), device=torch_device)
