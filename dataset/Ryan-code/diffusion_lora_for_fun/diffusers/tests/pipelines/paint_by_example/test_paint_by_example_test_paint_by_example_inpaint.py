def test_paint_by_example_inpaint(self):
    components = self.get_dummy_components()
    pipe = PaintByExamplePipeline(**components)
    pipe = pipe.to('cpu')
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    output = pipe(**inputs)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.4686, 0.5687, 0.4007, 0.5218, 0.5741, 
        0.4482, 0.494, 0.4629, 0.4503])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
