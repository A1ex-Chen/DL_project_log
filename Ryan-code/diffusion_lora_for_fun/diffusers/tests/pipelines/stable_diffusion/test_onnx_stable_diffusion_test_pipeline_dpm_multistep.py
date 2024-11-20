def test_pipeline_dpm_multistep(self):
    pipe = OnnxStableDiffusionPipeline.from_pretrained(self.hub_checkpoint,
        provider='CPUExecutionProvider')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler
        .config)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 128, 128, 3)
    expected_slice = np.array([0.53895, 0.60808, 0.47933, 0.49608, 0.51886,
        0.4995, 0.48053, 0.38957, 0.442])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
