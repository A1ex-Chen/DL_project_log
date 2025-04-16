@parameterized.expand([[33, [-0.034, 0.287, 0.1698, -0.0105, -0.3448, 
    0.3529, -0.1321, 0.1097], [-0.0344, 0.2912, 0.1687, -0.0137, -0.3462, 
    0.3552, -0.1337, 0.1078]], [47, [0.4397, 0.055, 0.2873, 0.2946, 0.0567,
    0.0855, -0.158, 0.2531], [0.4397, 0.055, 0.2873, 0.2946, 0.0567, 0.0855,
    -0.158, 0.2531]]])
def test_stable_diffusion_mode(self, seed, expected_slice, expected_slice_mps):
    model = self.get_sd_vae_model()
    image = self.get_sd_image(seed)
    with torch.no_grad():
        sample = model(image).sample
    assert sample.shape == image.shape
    output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
    expected_output_slice = torch.tensor(expected_slice_mps if torch_device ==
        'mps' else expected_slice)
    assert torch_all_close(output_slice, expected_output_slice, atol=0.003)
