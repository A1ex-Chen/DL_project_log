def test_less_downloads_passed_object(self):
    with tempfile.TemporaryDirectory() as tmpdirname:
        cached_folder = DiffusionPipeline.download(
            'hf-internal-testing/tiny-stable-diffusion-pipe',
            safety_checker=None, cache_dir=tmpdirname)
        assert 'safety_checker' not in os.listdir(cached_folder)
        assert 'unet' in os.listdir(cached_folder)
        assert 'tokenizer' in os.listdir(cached_folder)
        assert 'vae' in os.listdir(cached_folder)
        assert 'model_index.json' in os.listdir(cached_folder)
        assert 'scheduler' in os.listdir(cached_folder)
        assert 'feature_extractor' in os.listdir(cached_folder)
