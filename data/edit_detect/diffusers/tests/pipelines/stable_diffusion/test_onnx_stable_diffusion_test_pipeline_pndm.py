def test_pipeline_pndm(self):
    pipe = OnnxStableDiffusionPipeline.from_pretrained(self.hub_checkpoint,
        provider='CPUExecutionProvider')
    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config,
        skip_prk_steps=True)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 128, 128, 3)
    expected_slice = np.array([0.65863, 0.59425, 0.49326, 0.56313, 0.53875,
        0.56627, 0.51065, 0.39777, 0.4633])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
