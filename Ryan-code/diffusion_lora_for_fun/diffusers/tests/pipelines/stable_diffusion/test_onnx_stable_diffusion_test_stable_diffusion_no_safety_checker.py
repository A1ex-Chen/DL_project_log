def test_stable_diffusion_no_safety_checker(self):
    pipe = OnnxStableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', revision='onnx', safety_checker=
        None, feature_extractor=None, provider=self.gpu_provider,
        sess_options=self.gpu_options)
    assert isinstance(pipe, OnnxStableDiffusionPipeline)
    assert pipe.safety_checker is None
    image = pipe('example prompt', num_inference_steps=2).images[0]
    assert image is not None
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipe.save_pretrained(tmpdirname)
        pipe = OnnxStableDiffusionPipeline.from_pretrained(tmpdirname)
    assert pipe.safety_checker is None
    image = pipe('example prompt', num_inference_steps=2).images[0]
    assert image is not None
