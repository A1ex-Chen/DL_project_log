def test_single_file_components(self):
    controlnet = ControlNetModel.from_pretrained(
        'diffusers/controlnet-depth-sdxl-1.0', torch_dtype=torch.float16,
        variant='fp16')
    pipe = self.pipeline_class.from_pretrained(self.repo_id, variant='fp16',
        controlnet=controlnet, torch_dtype=torch.float16)
    pipe_single_file = self.pipeline_class.from_single_file(self.ckpt_path,
        controlnet=controlnet)
    super().test_single_file_components(pipe, pipe_single_file)
