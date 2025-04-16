def test_vae_image_processor_resize_pt(self):
    image_processor = VaeImageProcessor(do_resize=True, vae_scale_factor=1)
    input_pt = self.dummy_sample
    b, c, h, w = input_pt.shape
    scale = 2
    out_pt = image_processor.resize(image=input_pt, height=h // scale,
        width=w // scale)
    exp_pt_shape = b, c, h // scale, w // scale
    assert out_pt.shape == exp_pt_shape, f"resized image output shape '{out_pt.shape}' didn't match expected shape '{exp_pt_shape}'."
