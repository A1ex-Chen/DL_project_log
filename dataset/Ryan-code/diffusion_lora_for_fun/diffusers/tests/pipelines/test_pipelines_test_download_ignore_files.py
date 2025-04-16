def test_download_ignore_files(self):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = DiffusionPipeline.download(
            'hf-internal-testing/tiny-stable-diffusion-pipe-ignore-files')
        all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname))]
        files = [item for sublist in all_root_files for item in sublist]
        assert not any(f in ['vae/diffusion_pytorch_model.bin',
            'text_encoder/config.json'] for f in files)
        assert len(files) == 14
