def test_no_safetensors_download_when_doing_pytorch(self):
    use_safetensors = False
    with tempfile.TemporaryDirectory() as tmpdirname:
        _ = StableDiffusionPipeline.from_pretrained(
            'hf-internal-testing/diffusers-stable-diffusion-tiny-all',
            cache_dir=tmpdirname, use_safetensors=use_safetensors)
        path = os.path.join(tmpdirname,
            'models--hf-internal-testing--diffusers-stable-diffusion-tiny-all',
            'snapshots', '07838d72e12f9bcec1375b0482b80c1d399be843', 'unet')
        assert not os.path.exists(os.path.join(path,
            'diffusion_pytorch_model.safetensors'))
        assert os.path.exists(os.path.join(path, 'diffusion_pytorch_model.bin')
            )
