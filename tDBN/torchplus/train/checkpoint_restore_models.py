def restore_models(model_dir, models, global_step):
    name_to_model = _get_name_to_model_map(models)
    for name, model in name_to_model.items():
        ckpt_filename = '{}-{}.tckpt'.format(name, global_step)
        ckpt_path = model_dir + '/' + ckpt_filename
        restore(ckpt_path, model)
