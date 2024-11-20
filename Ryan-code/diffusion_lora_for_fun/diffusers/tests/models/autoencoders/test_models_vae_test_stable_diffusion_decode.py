@parameterized.expand([[13, [-0.0521, -0.2939, 0.154, -0.1855, -0.5936, -
    0.3138, -0.4579, -0.2275]], [37, [-0.182, -0.4345, -0.0455, -0.2923, -
    0.8035, -0.5089, -0.4795, -0.3106]]])
@require_torch_accelerator
@skip_mps
def test_stable_diffusion_decode(self, seed, expected_slice):
    model = self.get_sd_vae_model()
    encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64))
    with torch.no_grad():
        sample = model.decode(encoding).sample
    assert list(sample.shape) == [3, 3, 512, 512]
    output_slice = sample[-1, -2:, :2, -2:].flatten().cpu()
    expected_output_slice = torch.tensor(expected_slice)
    assert torch_all_close(output_slice, expected_output_slice, atol=0.002)
