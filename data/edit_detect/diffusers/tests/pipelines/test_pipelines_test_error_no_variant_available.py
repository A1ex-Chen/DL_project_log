def test_error_no_variant_available(self):
    variant = 'fp16'
    with self.assertRaises(ValueError) as error_context:
        _ = StableDiffusionPipeline.download(
            'hf-internal-testing/diffusers-stable-diffusion-tiny-all',
            variant=variant)
    assert 'but no such modeling files are available' in str(error_context.
        exception)
    assert variant in str(error_context.exception)
