@parameterized.expand([[33, [-0.3001, 0.0918, -2.6984, -3.972, -3.2099, -
    5.0353, 1.7338, -0.2065, 3.4267]], [47, [-1.503, -4.3871, -6.0355, -
    9.1157, -1.6661, -2.7853, 2.1607, -5.0823, 2.5633]]])
def test_stable_diffusion_encode_sample(self, seed, expected_slice):
    model = self.get_sd_vae_model()
    image = self.get_sd_image(seed)
    generator = self.get_generator(seed)
    with torch.no_grad():
        dist = model.encode(image).latent_dist
        sample = dist.sample(generator=generator)
    assert list(sample.shape) == [image.shape[0], 4] + [(i // 8) for i in
        image.shape[2:]]
    output_slice = sample[0, -1, -3:, -3:].flatten().cpu()
    expected_output_slice = torch.tensor(expected_slice)
    tolerance = 0.003 if torch_device != 'mps' else 0.01
    assert torch_all_close(output_slice, expected_output_slice, atol=tolerance)
