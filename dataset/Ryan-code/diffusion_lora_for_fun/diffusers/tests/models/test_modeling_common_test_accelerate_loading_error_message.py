def test_accelerate_loading_error_message(self):
    with self.assertRaises(ValueError) as error_context:
        UNet2DConditionModel.from_pretrained(
            'hf-internal-testing/stable-diffusion-broken', subfolder='unet')
    assert 'conv_out.bias' in str(error_context.exception)
