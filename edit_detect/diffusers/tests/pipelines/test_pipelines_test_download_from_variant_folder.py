def test_download_from_variant_folder(self):
    for use_safetensors in [False, True]:
        other_format = '.bin' if use_safetensors else '.safetensors'
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = StableDiffusionPipeline.download(
                'hf-internal-testing/stable-diffusion-all-variants',
                cache_dir=tmpdirname, use_safetensors=use_safetensors)
            all_root_files = [t[-1] for t in os.walk(tmpdirname)]
            files = [item for sublist in all_root_files for item in sublist]
            assert len(files
                ) == 15, f'We should only download 15 files, not {len(files)}'
            assert not any(f.endswith(other_format) for f in files)
            assert not any(len(f.split('.')) == 3 for f in files)
