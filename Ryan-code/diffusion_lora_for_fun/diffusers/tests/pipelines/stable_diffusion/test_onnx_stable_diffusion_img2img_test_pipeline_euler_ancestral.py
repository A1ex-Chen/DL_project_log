def test_pipeline_euler_ancestral(self):
    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(self.
        hub_checkpoint, provider='CPUExecutionProvider')
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.
        scheduler.config)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 128, 128, 3)
    expected_slice = np.array([0.52911, 0.60004, 0.49229, 0.49805, 0.54502,
        0.5068, 0.47777, 0.41028, 0.45304])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.1
