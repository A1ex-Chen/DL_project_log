def test_controlnet(self):
    model_repo = 'runwayml/stable-diffusion-v1-5'
    controlnet_repo = 'lllyasviel/sd-controlnet-canny'
    controlnet = ControlNetModel.from_pretrained(controlnet_repo,
        torch_dtype=torch.float16)
    pipe_txt2img = AutoPipelineForText2Image.from_pretrained(model_repo,
        controlnet=controlnet, torch_dtype=torch.float16)
    self.assertIsInstance(pipe_txt2img, AUTO_TEXT2IMAGE_PIPELINES_MAPPING[
        'stable-diffusion-controlnet'])
    pipe_img2img = AutoPipelineForImage2Image.from_pretrained(model_repo,
        controlnet=controlnet, torch_dtype=torch.float16)
    self.assertIsInstance(pipe_img2img, AUTO_IMAGE2IMAGE_PIPELINES_MAPPING[
        'stable-diffusion-controlnet'])
    pipe_inpaint = AutoPipelineForInpainting.from_pretrained(model_repo,
        controlnet=controlnet, torch_dtype=torch.float16)
    self.assertIsInstance(pipe_inpaint, AUTO_INPAINT_PIPELINES_MAPPING[
        'stable-diffusion-controlnet'])
    for pipe_from in [pipe_txt2img, pipe_img2img, pipe_inpaint]:
        pipe_to = AutoPipelineForText2Image.from_pipe(pipe_from)
        self.assertIsInstance(pipe_to, AUTO_TEXT2IMAGE_PIPELINES_MAPPING[
            'stable-diffusion-controlnet'])
        self.assertEqual(dict(pipe_to.config), dict(pipe_txt2img.config))
        pipe_to = AutoPipelineForImage2Image.from_pipe(pipe_from)
        self.assertIsInstance(pipe_to, AUTO_IMAGE2IMAGE_PIPELINES_MAPPING[
            'stable-diffusion-controlnet'])
        self.assertEqual(dict(pipe_to.config), dict(pipe_img2img.config))
        pipe_to = AutoPipelineForInpainting.from_pipe(pipe_from)
        self.assertIsInstance(pipe_to, AUTO_INPAINT_PIPELINES_MAPPING[
            'stable-diffusion-controlnet'])
        self.assertEqual(dict(pipe_to.config), dict(pipe_inpaint.config))
