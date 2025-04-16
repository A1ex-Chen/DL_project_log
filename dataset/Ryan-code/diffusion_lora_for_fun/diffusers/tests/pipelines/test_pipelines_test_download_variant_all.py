def test_download_variant_all(self):
    for use_safetensors in [False, True]:
        other_format = '.bin' if use_safetensors else '.safetensors'
        this_format = '.safetensors' if use_safetensors else '.bin'
        variant = 'fp16'
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = StableDiffusionPipeline.download(
                'hf-internal-testing/stable-diffusion-all-variants',
                cache_dir=tmpdirname, variant=variant, use_safetensors=
                use_safetensors)
            all_root_files = [t[-1] for t in os.walk(tmpdirname)]
            files = [item for sublist in all_root_files for item in sublist]
            assert len(files
                ) == 15, f'We should only download 15 files, not {len(files)}'
            assert len([f for f in files if f.endswith(
                f'{variant}{this_format}')]) == 4
            assert not any(f.endswith(this_format) and not f.endswith(
                f'{variant}{this_format}') for f in files)
            assert not any(f.endswith(other_format) for f in files)
