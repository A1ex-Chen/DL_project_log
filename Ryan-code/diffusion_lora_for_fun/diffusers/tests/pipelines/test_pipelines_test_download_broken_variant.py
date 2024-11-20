def test_download_broken_variant(self):
    for use_safetensors in [False, True]:
        for variant in [None, 'no_ema']:
            with self.assertRaises(OSError) as error_context:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    tmpdirname = StableDiffusionPipeline.from_pretrained(
                        'hf-internal-testing/stable-diffusion-broken-variants',
                        cache_dir=tmpdirname, variant=variant,
                        use_safetensors=use_safetensors)
            assert 'Error no file name' in str(error_context.exception)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = StableDiffusionPipeline.download(
                'hf-internal-testing/stable-diffusion-broken-variants',
                use_safetensors=use_safetensors, cache_dir=tmpdirname,
                variant='fp16')
            all_root_files = [t[-1] for t in os.walk(tmpdirname)]
            files = [item for sublist in all_root_files for item in sublist]
            assert len(files
                ) == 15, f'We should only download 15 files, not {len(files)}'
