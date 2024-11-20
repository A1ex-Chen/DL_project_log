def test_download_only_pytorch(self):
    with tempfile.TemporaryDirectory() as tmpdirname:
        _ = FlaxDiffusionPipeline.from_pretrained(
            'hf-internal-testing/tiny-stable-diffusion-pipe',
            safety_checker=None, cache_dir=tmpdirname)
        all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname,
            os.listdir(tmpdirname)[0], 'snapshots'))]
        files = [item for sublist in all_root_files for item in sublist]
        assert not any(f.endswith('.bin') for f in files)
