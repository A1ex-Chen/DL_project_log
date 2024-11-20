def test_pipeline_euler(self):
    pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(self.
        hub_checkpoint, provider='CPUExecutionProvider')
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.6974782, 0.68902093, 0.70135885, 0.7583618,
        0.7804545, 0.7854912, 0.78667426, 0.78743863, 0.78070223])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.1
