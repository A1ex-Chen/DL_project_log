def test_download_no_openvino_by_default(self):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = DiffusionPipeline.download(
            'hf-internal-testing/tiny-stable-diffusion-open-vino',
            cache_dir=tmpdirname)
        all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname))]
        files = [item for sublist in all_root_files for item in sublist]
        assert all(f.endswith('.json') or f.endswith('.bin') or f.endswith(
            '.txt') for f in files)
        assert not any('openvino_' in f for f in files)
