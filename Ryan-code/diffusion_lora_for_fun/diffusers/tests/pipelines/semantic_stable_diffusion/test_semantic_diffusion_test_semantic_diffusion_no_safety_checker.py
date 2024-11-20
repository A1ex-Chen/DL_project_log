def test_semantic_diffusion_no_safety_checker(self):
    pipe = StableDiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-lms-pipe',
        safety_checker=None)
    assert isinstance(pipe, StableDiffusionPipeline)
    assert isinstance(pipe.scheduler, LMSDiscreteScheduler)
    assert pipe.safety_checker is None
    image = pipe('example prompt', num_inference_steps=2).images[0]
    assert image is not None
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipe.save_pretrained(tmpdirname)
        pipe = StableDiffusionPipeline.from_pretrained(tmpdirname)
    assert pipe.safety_checker is None
    image = pipe('example prompt', num_inference_steps=2).images[0]
    assert image is not None
