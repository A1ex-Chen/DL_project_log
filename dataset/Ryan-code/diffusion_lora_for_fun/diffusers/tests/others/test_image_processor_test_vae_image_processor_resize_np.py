def test_vae_image_processor_resize_np(self):
    image_processor = VaeImageProcessor(do_resize=True, vae_scale_factor=1)
    input_pt = self.dummy_sample
    b, c, h, w = input_pt.shape
    scale = 2
    input_np = self.to_np(input_pt)
    out_np = image_processor.resize(image=input_np, height=h // scale,
        width=w // scale)
    exp_np_shape = b, h // scale, w // scale, c
    assert out_np.shape == exp_np_shape, f"resized image output shape '{out_np.shape}' didn't match expected shape '{exp_np_shape}'."
