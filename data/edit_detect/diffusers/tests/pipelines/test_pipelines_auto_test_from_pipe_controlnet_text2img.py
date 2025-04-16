def test_from_pipe_controlnet_text2img(self):
    pipe = AutoPipelineForText2Image.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-pipe')
    controlnet = ControlNetModel.from_pretrained(
        'hf-internal-testing/tiny-controlnet')
    pipe = AutoPipelineForText2Image.from_pipe(pipe, controlnet=controlnet)
    assert pipe.__class__.__name__ == 'StableDiffusionControlNetPipeline'
    assert 'controlnet' in pipe.components
    pipe = AutoPipelineForText2Image.from_pipe(pipe, controlnet=None)
    assert pipe.__class__.__name__ == 'StableDiffusionPipeline'
    assert 'controlnet' not in pipe.components
