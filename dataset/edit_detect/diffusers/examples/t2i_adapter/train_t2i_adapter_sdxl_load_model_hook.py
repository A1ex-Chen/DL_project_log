def load_model_hook(models, input_dir):
    while len(models) > 0:
        model = models.pop()
        load_model = T2IAdapter.from_pretrained(os.path.join(input_dir,
            't2iadapter'))
        if args.control_type != 'style':
            model.register_to_config(**load_model.config)
        model.load_state_dict(load_model.state_dict())
        del load_model
