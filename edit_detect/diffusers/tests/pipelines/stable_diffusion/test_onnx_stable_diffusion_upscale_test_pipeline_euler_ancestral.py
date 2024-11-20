def test_pipeline_euler_ancestral(self):
    pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(self.
        hub_checkpoint, provider='CPUExecutionProvider')
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.
        scheduler.config)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.77424496, 0.773601, 0.7645288, 0.7769598, 
        0.7772739, 0.7738688, 0.78187233, 0.77879584, 0.767043])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.1
