def test_single_file_components(self):
    model = self.model_class.from_pretrained(self.repo_id)
    model_single_file = self.model_class.from_single_file(self.ckpt_path,
        config=self.repo_id)
    PARAMS_TO_IGNORE = ['torch_dtype', '_name_or_path',
        '_use_default_values', '_diffusers_version']
    for param_name, param_value in model_single_file.config.items():
        if param_name in PARAMS_TO_IGNORE:
            continue
        assert model.config[param_name
            ] == param_value, f'{param_name} differs between pretrained loading and single file loading'
