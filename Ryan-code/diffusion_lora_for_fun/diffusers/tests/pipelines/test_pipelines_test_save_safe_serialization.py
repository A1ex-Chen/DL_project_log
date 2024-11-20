def test_save_safe_serialization(self):
    pipeline = StableDiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch')
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipeline.save_pretrained(tmpdirname, safe_serialization=True)
        vae_path = os.path.join(tmpdirname, 'vae',
            'diffusion_pytorch_model.safetensors')
        assert os.path.exists(vae_path), f'Could not find {vae_path}'
        _ = safetensors.torch.load_file(vae_path)
        unet_path = os.path.join(tmpdirname, 'unet',
            'diffusion_pytorch_model.safetensors')
        assert os.path.exists(unet_path), f'Could not find {unet_path}'
        _ = safetensors.torch.load_file(unet_path)
        text_encoder_path = os.path.join(tmpdirname, 'text_encoder',
            'model.safetensors')
        assert os.path.exists(text_encoder_path
            ), f'Could not find {text_encoder_path}'
        _ = safetensors.torch.load_file(text_encoder_path)
        pipeline = StableDiffusionPipeline.from_pretrained(tmpdirname)
        assert pipeline.unet is not None
        assert pipeline.vae is not None
        assert pipeline.text_encoder is not None
        assert pipeline.scheduler is not None
        assert pipeline.feature_extractor is not None
