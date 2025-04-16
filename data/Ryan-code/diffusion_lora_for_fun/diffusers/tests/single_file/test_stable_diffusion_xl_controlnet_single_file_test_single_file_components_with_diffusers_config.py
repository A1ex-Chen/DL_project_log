def test_single_file_components_with_diffusers_config(self):
    controlnet = ControlNetModel.from_pretrained(
        'diffusers/controlnet-depth-sdxl-1.0', torch_dtype=torch.float16,
        variant='fp16')
    pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=
        controlnet)
    pipe_single_file = self.pipeline_class.from_single_file(self.ckpt_path,
        controlnet=controlnet, config=self.repo_id)
    super()._compare_component_configs(pipe, pipe_single_file)
