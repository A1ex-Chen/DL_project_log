def test_pipeline_lms(self):
    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(self.
        hub_checkpoint, provider='CPUExecutionProvider')
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=None)
    _ = pipe(**self.get_dummy_inputs())
    inputs = self.get_dummy_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 128, 128, 3)
    expected_slice = np.array([0.52761, 0.59977, 0.49033, 0.49619, 0.54282,
        0.50311, 0.476, 0.40918, 0.45203])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.1
