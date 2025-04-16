@require_torch_accelerator
def test_from_pretrained_accelerate(self):
    model, _ = UNet2DModel.from_pretrained('fusing/unet-ldm-dummy-update',
        output_loading_info=True)
    model.to(torch_device)
    image = model(**self.dummy_input).sample
    assert image is not None, 'Make sure output is not None'
