@parameterized.expand([[27, [-0.0369, 0.0207, -0.0776, -0.0682, -0.1747, -
    0.193, -0.1465, -0.2039]], [16, [-0.1628, -0.2134, -0.2747, -0.2642, -
    0.3774, -0.4404, -0.3687, -0.4277]]])
@require_torch_accelerator_with_fp16
def test_stable_diffusion_decode_fp16(self, seed, expected_slice):
    model = self.get_sd_vae_model(fp16=True)
    encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64), fp16=True)
    with torch.no_grad():
        sample = model.decode(encoding).sample
    assert list(sample.shape) == [3, 3, 512, 512]
    output_slice = sample[-1, -2:, :2, -2:].flatten().float().cpu()
    expected_output_slice = torch.tensor(expected_slice)
    assert torch_all_close(output_slice, expected_output_slice, atol=0.005)
