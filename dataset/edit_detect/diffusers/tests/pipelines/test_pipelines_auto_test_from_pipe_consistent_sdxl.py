def test_from_pipe_consistent_sdxl(self):
    pipe = AutoPipelineForImage2Image.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-xl-pipe',
        requires_aesthetics_score=True, force_zeros_for_empty_prompt=False)
    original_config = dict(pipe.config)
    pipe = AutoPipelineForText2Image.from_pipe(pipe)
    pipe = AutoPipelineForImage2Image.from_pipe(pipe)
    assert dict(pipe.config) == original_config
