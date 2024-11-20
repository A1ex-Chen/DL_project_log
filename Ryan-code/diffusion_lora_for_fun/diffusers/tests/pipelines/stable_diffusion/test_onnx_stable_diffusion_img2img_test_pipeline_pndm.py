def test_pipeline_pndm(self):
    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(self.
        hub_checkpoint, provider='CPUExecutionProvider')
    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config,
        skip_prk_steps=True)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 128, 128, 3)
    expected_slice = np.array([0.61737, 0.54642, 0.53183, 0.54465, 0.52742,
        0.60525, 0.49969, 0.40655, 0.48154])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.1
