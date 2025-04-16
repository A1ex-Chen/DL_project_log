def test_weight_overwrite(self):
    with tempfile.TemporaryDirectory() as tmpdirname, self.assertRaises(
        ValueError) as error_context:
        UNet2DConditionModel.from_pretrained(
            'hf-internal-testing/tiny-stable-diffusion-torch', subfolder=
            'unet', cache_dir=tmpdirname, in_channels=9)
    assert 'Cannot load' in str(error_context.exception)
    with tempfile.TemporaryDirectory() as tmpdirname:
        model = UNet2DConditionModel.from_pretrained(
            'hf-internal-testing/tiny-stable-diffusion-torch', subfolder=
            'unet', cache_dir=tmpdirname, in_channels=9, low_cpu_mem_usage=
            False, ignore_mismatched_sizes=True)
    assert model.config.in_channels == 9
