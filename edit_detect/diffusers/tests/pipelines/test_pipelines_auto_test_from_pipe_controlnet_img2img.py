def test_from_pipe_controlnet_img2img(self):
    pipe = AutoPipelineForImage2Image.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-pipe')
    controlnet = ControlNetModel.from_pretrained(
        'hf-internal-testing/tiny-controlnet')
    pipe = AutoPipelineForImage2Image.from_pipe(pipe, controlnet=controlnet)
    assert pipe.__class__.__name__ == 'StableDiffusionControlNetImg2ImgPipeline'
    assert 'controlnet' in pipe.components
    pipe = AutoPipelineForImage2Image.from_pipe(pipe, controlnet=None)
    assert pipe.__class__.__name__ == 'StableDiffusionImg2ImgPipeline'
    assert 'controlnet' not in pipe.components
