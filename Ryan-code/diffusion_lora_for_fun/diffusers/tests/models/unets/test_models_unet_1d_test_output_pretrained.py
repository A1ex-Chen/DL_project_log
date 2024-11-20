def test_output_pretrained(self):
    value_function, vf_loading_info = UNet1DModel.from_pretrained(
        'bglick13/hopper-medium-v2-value-function-hor32',
        output_loading_info=True, subfolder='value_function')
    torch.manual_seed(0)
    backend_manual_seed(torch_device, 0)
    num_features = value_function.config.in_channels
    seq_len = 14
    noise = torch.randn((1, seq_len, num_features)).permute(0, 2, 1)
    time_step = torch.full((num_features,), 0)
    with torch.no_grad():
        output = value_function(noise, time_step).sample
    expected_output_slice = torch.tensor([165.25] * seq_len)
    self.assertTrue(torch.allclose(output, expected_output_slice, rtol=0.001))
