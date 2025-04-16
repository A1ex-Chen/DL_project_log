def test_pipeline_default_ddim(self):
    pipe = OnnxStableDiffusionPipeline.from_pretrained(self.hub_checkpoint,
        provider='CPUExecutionProvider')
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 128, 128, 3)
    expected_slice = np.array([0.65072, 0.58492, 0.48219, 0.55521, 0.5318, 
        0.55939, 0.50697, 0.398, 0.46455])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
