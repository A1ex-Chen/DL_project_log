def param_init_fn(self, module):
    init_fn_name = self.config.init_config['name']
    MODEL_INIT_REGISTRY[init_fn_name](module=module, n_layers=self.config.
        n_layers, d_model=self.config.d_model, **self.config.init_config)
