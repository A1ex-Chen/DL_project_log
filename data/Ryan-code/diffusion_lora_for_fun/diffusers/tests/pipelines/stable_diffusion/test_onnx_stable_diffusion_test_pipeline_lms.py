def test_pipeline_lms(self):
    pipe = OnnxStableDiffusionPipeline.from_pretrained(self.hub_checkpoint,
        provider='CPUExecutionProvider')
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 128, 128, 3)
    expected_slice = np.array([0.53755, 0.60786, 0.47402, 0.49488, 0.51869,
        0.49819, 0.47985, 0.38957, 0.44279])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
