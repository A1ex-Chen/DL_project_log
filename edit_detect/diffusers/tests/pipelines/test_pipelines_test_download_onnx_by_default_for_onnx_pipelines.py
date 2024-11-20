@require_onnxruntime
def test_download_onnx_by_default_for_onnx_pipelines(self):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = DiffusionPipeline.download(
            'hf-internal-testing/tiny-random-OnnxStableDiffusionPipeline',
            cache_dir=tmpdirname)
        all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname))]
        files = [item for sublist in all_root_files for item in sublist]
        assert any(f.endswith('.json') or f.endswith('.bin') or f.endswith(
            '.txt') for f in files)
        assert any(f.endswith('.onnx') for f in files)
        assert any(f.endswith('.pb') for f in files)
