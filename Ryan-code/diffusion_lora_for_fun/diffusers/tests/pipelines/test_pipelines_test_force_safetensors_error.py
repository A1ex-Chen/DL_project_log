def test_force_safetensors_error(self):
    with tempfile.TemporaryDirectory() as tmpdirname:
        with self.assertRaises(EnvironmentError):
            tmpdirname = DiffusionPipeline.download(
                'hf-internal-testing/tiny-stable-diffusion-pipe-no-safetensors'
                , safety_checker=None, cache_dir=tmpdirname,
                use_safetensors=True)
