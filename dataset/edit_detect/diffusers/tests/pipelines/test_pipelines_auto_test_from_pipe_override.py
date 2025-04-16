def test_from_pipe_override(self):
    pipe = AutoPipelineForText2Image.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-pipe',
        requires_safety_checker=False)
    pipe = AutoPipelineForImage2Image.from_pipe(pipe,
        requires_safety_checker=True)
    assert pipe.config.requires_safety_checker is True
    pipe = AutoPipelineForText2Image.from_pipe(pipe,
        requires_safety_checker=True)
    assert pipe.config.requires_safety_checker is True
