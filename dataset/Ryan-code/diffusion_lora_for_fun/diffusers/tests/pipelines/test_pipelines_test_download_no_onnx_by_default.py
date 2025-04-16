def test_download_no_onnx_by_default(self):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = DiffusionPipeline.download(
            'hf-internal-testing/tiny-stable-diffusion-xl-pipe', cache_dir=
            tmpdirname, use_safetensors=False)
        all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname))]
        files = [item for sublist in all_root_files for item in sublist]
        assert all(f.endswith('.json') or f.endswith('.bin') or f.endswith(
            '.txt') for f in files)
        assert not any(f.endswith('.onnx') or f.endswith('.pb') for f in files)
