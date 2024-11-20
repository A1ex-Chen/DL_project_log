def test_single_file_components(self):
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/control_v11p_sd15_canny')
    pipe = self.pipeline_class.from_pretrained(self.repo_id, variant='fp16',
        safety_checker=None, controlnet=controlnet)
    pipe_single_file = self.pipeline_class.from_single_file(self.ckpt_path,
        safety_checker=None, controlnet=controlnet)
    super()._compare_component_configs(pipe, pipe_single_file)
