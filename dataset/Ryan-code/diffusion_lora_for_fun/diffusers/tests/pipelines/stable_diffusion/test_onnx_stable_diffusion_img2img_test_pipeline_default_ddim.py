def test_pipeline_default_ddim(self):
    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(self.
        hub_checkpoint, provider='CPUExecutionProvider')
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 128, 128, 3)
    expected_slice = np.array([0.69643, 0.58484, 0.50314, 0.5876, 0.55368, 
        0.59643, 0.51529, 0.41217, 0.49087])
    assert np.abs(image_slice - expected_slice).max() < 0.1
