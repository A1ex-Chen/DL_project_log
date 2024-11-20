@parameterized.expand([(True,), (False,)])
def test_tae_roundtrip(self, enable_tiling):
    model = self.get_sd_vae_model()
    if enable_tiling:
        model.enable_tiling()
    image = -torch.ones(1, 3, 1024, 1024, device=torch_device)
    image[..., 256:768, 256:768] = 1.0
    with torch.no_grad():
        sample = model(image).sample

    def downscale(x):
        return torch.nn.functional.avg_pool2d(x, model.spatial_scale_factor)
    assert torch_all_close(downscale(sample), downscale(image), atol=0.125)
