def test_single_file_components_with_original_config(self):
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/control_v11p_sd15_canny', variant='fp16')
    pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=
        controlnet)
    pipe_single_file = self.pipeline_class.from_single_file(self.ckpt_path,
        controlnet=controlnet, original_config=self.original_config)
    super()._compare_component_configs(pipe, pipe_single_file)
