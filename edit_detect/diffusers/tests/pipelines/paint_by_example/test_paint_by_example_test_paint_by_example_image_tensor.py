def test_paint_by_example_image_tensor(self):
    device = 'cpu'
    inputs = self.get_dummy_inputs()
    inputs.pop('mask_image')
    image = self.convert_to_pt(inputs.pop('image'))
    mask_image = image.clamp(0, 1) / 2
    pipe = PaintByExamplePipeline(**self.get_dummy_components())
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    output = pipe(image=image, mask_image=mask_image[:, 0], **inputs)
    out_1 = output.images
    image = image.cpu().permute(0, 2, 3, 1)[0]
    mask_image = mask_image.cpu().permute(0, 2, 3, 1)[0]
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    mask_image = Image.fromarray(np.uint8(mask_image)).convert('RGB')
    output = pipe(**self.get_dummy_inputs())
    out_2 = output.images
    assert out_1.shape == (1, 64, 64, 3)
    assert np.abs(out_1.flatten() - out_2.flatten()).max() < 0.05
