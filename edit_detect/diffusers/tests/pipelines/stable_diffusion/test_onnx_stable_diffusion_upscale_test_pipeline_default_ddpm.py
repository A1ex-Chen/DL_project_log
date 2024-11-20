def test_pipeline_default_ddpm(self):
    pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(self.
        hub_checkpoint, provider='CPUExecutionProvider')
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.6957, 0.7002, 0.7186, 0.6881, 0.6693, 
        0.691, 0.7445, 0.7274, 0.7056])
    assert np.abs(image_slice - expected_slice).max() < 0.1
