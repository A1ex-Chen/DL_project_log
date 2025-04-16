def restore_latest_checkpoints(model_dir, models):
    name_to_model = _get_name_to_model_map(models)
    for name, model in name_to_model.items():
        latest_ckpt = latest_checkpoint(model_dir, name)
        if latest_ckpt is not None:
            restore(latest_ckpt, model)
        else:
            raise ValueError("model {}'s ckpt isn't exist".format(name))
