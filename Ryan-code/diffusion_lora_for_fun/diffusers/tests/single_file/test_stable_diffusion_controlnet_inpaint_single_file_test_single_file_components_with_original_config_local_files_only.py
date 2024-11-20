def test_single_file_components_with_original_config_local_files_only(self):
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/control_v11p_sd15_canny', torch_dtype=torch.float16,
        variant='fp16')
    pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=
        controlnet, safety_checker=None)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_filename = self.ckpt_path.split('/')[-1]
        local_ckpt_path = download_single_file_checkpoint(self.repo_id,
            ckpt_filename, tmpdir)
        local_original_config = download_original_config(self.
            original_config, tmpdir)
        pipe_single_file = self.pipeline_class.from_single_file(local_ckpt_path
            , original_config=local_original_config, controlnet=controlnet,
            safety_checker=None, local_files_only=True)
    super()._compare_component_configs(pipe, pipe_single_file)
