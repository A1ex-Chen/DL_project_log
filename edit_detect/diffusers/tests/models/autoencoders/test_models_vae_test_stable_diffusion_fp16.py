@parameterized.expand([[33, [-0.0513, 0.0289, 1.3799, 0.2166, -0.2573, -
    0.0871, 0.5103, -0.0999]], [47, [-0.4128, -0.132, -0.3704, 0.1965, -
    0.4116, -0.2332, -0.334, 0.2247]]])
@require_torch_accelerator_with_fp16
def test_stable_diffusion_fp16(self, seed, expected_slice):
    model = self.get_sd_vae_model(fp16=True)
    image = self.get_sd_image(seed, fp16=True)
    generator = self.get_generator(seed)
    with torch.no_grad():
        sample = model(image, generator=generator, sample_posterior=True
            ).sample
    assert sample.shape == image.shape
    output_slice = sample[-1, -2:, :2, -2:].flatten().float().cpu()
    expected_output_slice = torch.tensor(expected_slice)
    assert torch_all_close(output_slice, expected_output_slice, atol=0.01)
