def instantiate_cond_stage(self, config):
    if not self.cond_stage_trainable:
        if config == '__is_first_stage__':
            print('Using first stage also as cond stage.')
            self.cond_stage_model = self.first_stage_model
        elif config == '__is_unconditional__':
            print(
                f'Training {self.__class__.__name__} as an unconditional model.'
                )
            self.cond_stage_model = None
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
    else:
        assert config != '__is_first_stage__'
        assert config != '__is_unconditional__'
        model = instantiate_from_config(config)
        self.cond_stage_model = model
    self.cond_stage_model = self.cond_stage_model.to(self.device)
