def test_pipeline_dpm_multistep(self):
    pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(self.
        hub_checkpoint, provider='CPUExecutionProvider')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler
        .config)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.7659278, 0.76437664, 0.75579107, 0.7691116,
        0.77666986, 0.7727672, 0.7758664, 0.7812226, 0.76942515])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.1
