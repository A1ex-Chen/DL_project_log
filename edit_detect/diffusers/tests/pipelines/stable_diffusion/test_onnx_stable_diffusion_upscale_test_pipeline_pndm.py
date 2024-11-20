def test_pipeline_pndm(self):
    pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(self.
        hub_checkpoint, provider='CPUExecutionProvider')
    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config,
        skip_prk_steps=True)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.7349, 0.7347, 0.7034, 0.7696, 0.7876, 
        0.7597, 0.7916, 0.8085, 0.8036])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.1
