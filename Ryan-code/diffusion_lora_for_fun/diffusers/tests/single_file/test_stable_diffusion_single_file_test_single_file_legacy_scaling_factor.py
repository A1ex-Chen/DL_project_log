def test_single_file_legacy_scaling_factor(self):
    new_scaling_factor = 10.0
    init_pipe = self.pipeline_class.from_single_file(self.ckpt_path)
    pipe = self.pipeline_class.from_single_file(self.ckpt_path,
        scaling_factor=new_scaling_factor)
    assert init_pipe.vae.config.scaling_factor != new_scaling_factor
    assert pipe.vae.config.scaling_factor == new_scaling_factor
