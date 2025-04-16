def test_from_pipe_controlnet_new_task(self):
    pipe_text2img = AutoPipelineForText2Image.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch')
    controlnet = ControlNetModel.from_pretrained(
        'hf-internal-testing/tiny-controlnet')
    pipe_control_img2img = AutoPipelineForImage2Image.from_pipe(pipe_text2img,
        controlnet=controlnet)
    assert pipe_control_img2img.__class__.__name__ == 'StableDiffusionControlNetImg2ImgPipeline'
    assert 'controlnet' in pipe_control_img2img.components
    pipe_inpaint = AutoPipelineForInpainting.from_pipe(pipe_control_img2img,
        controlnet=None)
    assert pipe_inpaint.__class__.__name__ == 'StableDiffusionInpaintPipeline'
    assert 'controlnet' not in pipe_inpaint.components
    pipe_control_text2img = AutoPipelineForText2Image.from_pipe(
        pipe_control_img2img)
    assert pipe_control_text2img.__class__.__name__ == 'StableDiffusionControlNetPipeline'
    assert 'controlnet' in pipe_control_text2img.components
    pipe_control_text2img = AutoPipelineForText2Image.from_pipe(
        pipe_control_img2img, controlnet=controlnet)
    assert pipe_control_text2img.__class__.__name__ == 'StableDiffusionControlNetPipeline'
    assert 'controlnet' in pipe_control_text2img.components
    pipe_control_text2img = AutoPipelineForText2Image.from_pipe(
        pipe_control_text2img, controlnet=controlnet)
    assert pipe_control_text2img.__class__.__name__ == 'StableDiffusionControlNetPipeline'
    assert 'controlnet' in pipe_control_text2img.components
    pipe_control_inpaint = AutoPipelineForInpainting.from_pipe(
        pipe_control_img2img)
    assert pipe_control_inpaint.__class__.__name__ == 'StableDiffusionControlNetInpaintPipeline'
    assert 'controlnet' in pipe_control_inpaint.components
    pipe_control_inpaint = AutoPipelineForInpainting.from_pipe(
        pipe_control_img2img, controlnet=controlnet)
    assert pipe_control_inpaint.__class__.__name__ == 'StableDiffusionControlNetInpaintPipeline'
    assert 'controlnet' in pipe_control_inpaint.components
    pipe_control_inpaint = AutoPipelineForInpainting.from_pipe(
        pipe_control_inpaint, controlnet=controlnet)
    assert pipe_control_inpaint.__class__.__name__ == 'StableDiffusionControlNetInpaintPipeline'
    assert 'controlnet' in pipe_control_inpaint.components
    pipe_control_img2img = AutoPipelineForImage2Image.from_pipe(
        pipe_control_text2img)
    assert pipe_control_img2img.__class__.__name__ == 'StableDiffusionControlNetImg2ImgPipeline'
    assert 'controlnet' in pipe_control_img2img.components
    pipe_control_img2img = AutoPipelineForImage2Image.from_pipe(
        pipe_control_text2img, controlnet=controlnet)
    assert pipe_control_img2img.__class__.__name__ == 'StableDiffusionControlNetImg2ImgPipeline'
    assert 'controlnet' in pipe_control_img2img.components
    pipe_control_img2img = AutoPipelineForImage2Image.from_pipe(
        pipe_control_img2img, controlnet=controlnet)
    assert pipe_control_img2img.__class__.__name__ == 'StableDiffusionControlNetImg2ImgPipeline'
    assert 'controlnet' in pipe_control_img2img.components
