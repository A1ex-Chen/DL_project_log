def load_model_hook(models, input_dir):
    while len(models) > 0:
        model = models.pop()
        load_model = ControlNetModel.from_pretrained(input_dir, subfolder=
            'controlnet')
        model.register_to_config(**load_model.config)
        model.load_state_dict(load_model.state_dict())
        del load_model
