def _create_and_check_torchscript(self, config, inputs_dict):
    if not self.test_torchscript:
        return
    configs_no_init = _config_zero_init(config)
    configs_no_init.torchscript = True
    for model_class in self.all_model_classes:
        model = model_class(config=configs_no_init)
        model.eval()
        inputs = inputs_dict['input_ids']
        try:
            torch.jit.trace(model, inputs)
        except RuntimeError:
            self.fail("Couldn't trace module.")
        try:
            traced_gpt2 = torch.jit.trace(model, inputs)
            torch.jit.save(traced_gpt2, 'traced_model.pt')
        except RuntimeError:
            self.fail("Couldn't save module.")
        try:
            loaded_model = torch.jit.load('traced_model.pt')
            os.remove('traced_model.pt')
        except ValueError:
            self.fail("Couldn't load module.")
        model.eval()
        loaded_model.eval()
        model_params = model.parameters()
        loaded_model_params = loaded_model.parameters()
        models_equal = True
        for p1, p2 in zip(model_params, loaded_model_params):
            if p1.data.ne(p2.data).sum() > 0:
                models_equal = False
        self.assertTrue(models_equal)
