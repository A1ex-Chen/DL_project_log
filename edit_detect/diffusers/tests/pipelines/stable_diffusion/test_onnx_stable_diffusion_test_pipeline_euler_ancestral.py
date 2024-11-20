def test_pipeline_euler_ancestral(self):
    pipe = OnnxStableDiffusionPipeline.from_pretrained(self.hub_checkpoint,
        provider='CPUExecutionProvider')
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.
        scheduler.config)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 128, 128, 3)
    expected_slice = np.array([0.53817, 0.60812, 0.47384, 0.4953, 0.51894, 
        0.49814, 0.47984, 0.38958, 0.44271])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
