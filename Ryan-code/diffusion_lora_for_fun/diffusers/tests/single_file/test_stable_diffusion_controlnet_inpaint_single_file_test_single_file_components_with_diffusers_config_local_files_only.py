def test_single_file_components_with_diffusers_config_local_files_only(self):
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/control_v11p_sd15_canny', torch_dtype=torch.float16,
        variant='fp16')
    pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=
        controlnet, safety_checker=None)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_filename = self.ckpt_path.split('/')[-1]
        local_ckpt_path = download_single_file_checkpoint(self.repo_id,
            ckpt_filename, tmpdir)
        local_diffusers_config = download_diffusers_config(self.repo_id, tmpdir
            )
        pipe_single_file = self.pipeline_class.from_single_file(local_ckpt_path
            , config=local_diffusers_config, controlnet=controlnet,
            safety_checker=None, local_files_only=True)
    super()._compare_component_configs(pipe, pipe_single_file)
