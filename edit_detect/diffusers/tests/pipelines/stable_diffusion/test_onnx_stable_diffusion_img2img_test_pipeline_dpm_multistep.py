def test_pipeline_dpm_multistep(self):
    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(self.
        hub_checkpoint, provider='CPUExecutionProvider')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler
        .config)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 128, 128, 3)
    expected_slice = np.array([0.65331, 0.58277, 0.48204, 0.56059, 0.53665,
        0.56235, 0.50969, 0.40009, 0.46552])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.1
