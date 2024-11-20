def test_no_pytorch_download_when_doing_safetensors(self):
    with tempfile.TemporaryDirectory() as tmpdirname:
        _ = StableDiffusionPipeline.from_pretrained(
            'hf-internal-testing/diffusers-stable-diffusion-tiny-all',
            cache_dir=tmpdirname)
        path = os.path.join(tmpdirname,
            'models--hf-internal-testing--diffusers-stable-diffusion-tiny-all',
            'snapshots', '07838d72e12f9bcec1375b0482b80c1d399be843', 'unet')
        assert os.path.exists(os.path.join(path,
            'diffusion_pytorch_model.safetensors'))
        assert not os.path.exists(os.path.join(path,
            'diffusion_pytorch_model.bin'))
