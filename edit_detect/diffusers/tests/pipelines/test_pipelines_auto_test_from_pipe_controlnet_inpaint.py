def test_from_pipe_controlnet_inpaint(self):
    pipe = AutoPipelineForInpainting.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch')
    controlnet = ControlNetModel.from_pretrained(
        'hf-internal-testing/tiny-controlnet')
    pipe = AutoPipelineForInpainting.from_pipe(pipe, controlnet=controlnet)
    assert pipe.__class__.__name__ == 'StableDiffusionControlNetInpaintPipeline'
    assert 'controlnet' in pipe.components
    pipe = AutoPipelineForInpainting.from_pipe(pipe, controlnet=None)
    assert pipe.__class__.__name__ == 'StableDiffusionInpaintPipeline'
    assert 'controlnet' not in pipe.components
