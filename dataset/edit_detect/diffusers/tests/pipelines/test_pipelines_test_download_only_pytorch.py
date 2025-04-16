def test_download_only_pytorch(self):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = DiffusionPipeline.download(
            'hf-internal-testing/tiny-stable-diffusion-pipe',
            safety_checker=None, cache_dir=tmpdirname)
        all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname))]
        files = [item for sublist in all_root_files for item in sublist]
        assert not any(f.endswith('.msgpack') for f in files)
        assert not any(f.endswith('.safetensors') for f in files)
