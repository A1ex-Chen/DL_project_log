def test_single_file_components_local_files_only(self):
    controlnet = ControlNetModel.from_pretrained(
        'diffusers/controlnet-depth-sdxl-1.0', torch_dtype=torch.float16,
        variant='fp16')
    pipe = self.pipeline_class.from_pretrained(self.repo_id, variant='fp16',
        controlnet=controlnet, torch_dtype=torch.float16)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_filename = self.ckpt_path.split('/')[-1]
        local_ckpt_path = download_single_file_checkpoint(self.repo_id,
            ckpt_filename, tmpdir)
        single_file_pipe = self.pipeline_class.from_single_file(local_ckpt_path
            , controlnet=controlnet, safety_checker=None, local_files_only=True
            )
    self._compare_component_configs(pipe, single_file_pipe)
