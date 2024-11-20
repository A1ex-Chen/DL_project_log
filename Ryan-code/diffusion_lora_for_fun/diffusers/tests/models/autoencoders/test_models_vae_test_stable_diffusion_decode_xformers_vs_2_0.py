@parameterized.expand([(13,), (16,), (37,)])
@require_torch_gpu
@unittest.skipIf(not is_xformers_available(), reason=
    'xformers is not required when using PyTorch 2.0.')
def test_stable_diffusion_decode_xformers_vs_2_0(self, seed):
    model = self.get_sd_vae_model()
    encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64))
    with torch.no_grad():
        sample = model.decode(encoding).sample
    model.enable_xformers_memory_efficient_attention()
    with torch.no_grad():
        sample_2 = model.decode(encoding).sample
    assert list(sample.shape) == [3, 3, 512, 512]
    assert torch_all_close(sample, sample_2, atol=0.05)
