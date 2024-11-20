@parameterized.expand([[33, [-0.0344, 0.2912, 0.1687, -0.0137, -0.3462, 
    0.3552, -0.1337, 0.1078], [-0.1603, 0.9878, -0.0495, -0.079, -0.2709, 
    0.8375, -0.206, -0.0824]], [47, [0.44, 0.0543, 0.2873, 0.2946, 0.0553, 
    0.0839, -0.1585, 0.2529], [-0.2376, 0.1168, 0.1332, -0.484, -0.2508, -
    0.0791, -0.0493, -0.4089]]])
def test_stable_diffusion(self, seed, expected_slice, expected_slice_mps):
    model = self.get_sd_vae_model()
    image = self.get_sd_image(seed)
    generator = self.get_generator(seed)
    with torch.no_grad():
        sample = model(image, generator=generator, sample_posterior=True
            ).sample
    assert sample.shape == image.shape
    output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
    expected_output_slice = torch.tensor(expected_slice_mps if torch_device ==
        'mps' else expected_slice)
    assert torch_all_close(output_slice, expected_output_slice, atol=0.005)
