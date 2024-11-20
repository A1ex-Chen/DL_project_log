def test_download_safetensors_index(self):
    for variant in ['fp16', None]:
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = DiffusionPipeline.download(
                'hf-internal-testing/tiny-stable-diffusion-pipe-indexes',
                cache_dir=tmpdirname, use_safetensors=True, variant=variant)
            all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname))]
            files = [item for sublist in all_root_files for item in sublist]
            if variant is None:
                assert not any('fp16' in f for f in files)
            else:
                model_files = [f for f in files if 'safetensors' in f]
                assert all('fp16' in f for f in model_files)
            assert len([f for f in files if '.safetensors' in f]) == 8
            assert not any('.bin' in f for f in files)
